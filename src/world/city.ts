import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { DISTRICTS, type DistrictId, type View } from '../sim/architecture'
import { executionPath, type Kernel, type Precision, type sampleKernel } from '../sim/model'

type Sample = ReturnType<typeof sampleKernel>
interface DistrictView { id: DistrictId; root: THREE.Group; label: HTMLButtonElement; bounds: THREE.Box3; outline: THREE.Box3Helper }
interface Flow { curve: THREE.CurvePath<THREE.Vector3>; packets: THREE.Mesh[]; engine: 'memory' | 'command' | 'cuda' | 'tensor' }

export function createCity(host: HTMLDivElement, onSelect: (id: DistrictId) => void) {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color('#151819')
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 220)
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.domElement.tabIndex = 0
  renderer.domElement.setAttribute('aria-label', 'Interactive CUDA architecture city')
  host.append(renderer.domElement)
  const labels = document.createElement('div'); labels.className = 'world-labels'; host.append(labels)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.enablePan = false
  controls.minDistance = 8
  controls.maxDistance = 100
  controls.maxPolarAngle = Math.PI * 0.46
  const ambient = new THREE.HemisphereLight(0xf4fcff, 0x414032, 2.5)
  const sun = new THREE.DirectionalLight(0xfff5df, 3.2)
  sun.position.set(-14, 25, 12); sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  Object.assign(sun.shadow.camera, { left: -24, right: 24, top: 22, bottom: -22 })
  sun.shadow.normalBias = 0.04
  scene.add(ambient, sun)
  const world = new THREE.Group(); scene.add(world)
  const cube = new THREE.BoxGeometry(1, 1, 1)
  const materials = new Map<string, THREE.MeshStandardMaterial>()
  const transient: THREE.BufferGeometry[] = []
  const districts: DistrictView[] = []
  const flows: Flow[] = []
  const warpMeshes: THREE.InstancedMesh[] = []
  let view: View = 'gpu'
  let selected: DistrictId = 'sm'
  let showLabels = true
  let flight: { target: THREE.Vector3; position: THREE.Vector3 } | null = null
  const reduced = matchMedia('(prefers-reduced-motion: reduce)')

  function material(color: string) {
    if (!materials.has(color)) materials.set(color, new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.2 }))
    return materials.get(color)!
  }
  function box(parent: THREE.Object3D, size: [number, number, number], position: [number, number, number], color: string) {
    const mesh = new THREE.Mesh(cube, material(color))
    mesh.scale.set(...size); mesh.position.set(...position)
    mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh)
    return mesh
  }
  function bank(parent: THREE.Object3D, position: [number, number, number], color: string, columns = 6) {
    const group = new THREE.Group(); group.position.set(...position); parent.add(group)
    box(group, [columns * 0.5 + 0.4, 0.35, 1.2], [0, 0.2, 0], '#485454')
    for (let index = 0; index < columns; index += 1) box(group, [0.32, 0.85, 0.8], [(index - (columns - 1) / 2) * 0.5, 0.8, 0], color)
    return group
  }
  function node(id: DistrictId, position: [number, number, number], build: (group: THREE.Group) => void) {
    const definition = DISTRICTS.find((district) => district.id === id)!
    const root = new THREE.Group(); root.position.set(...position); root.userData.districtId = id; world.add(root)
    build(root)
    root.updateWorldMatrix(true, true)
    const bounds = new THREE.Box3().setFromObject(root)
    const outline = new THREE.Box3Helper(bounds.clone().expandByScalar(0.15), new THREE.Color('#def4b8'))
    outline.visible = id === selected; world.add(outline)
    const label = document.createElement('button')
    label.className = 'world-label'; label.dataset.district = id; label.style.setProperty('--district', definition.color)
    label.setAttribute('aria-label', `Inspect ${definition.name}`)
    label.title = definition.subtitle
    const title = document.createElement('strong'); title.textContent = definition.name
    const subtitle = document.createElement('span'); subtitle.textContent = definition.subtitle
    label.append(title, subtitle); labels.append(label)
    label.addEventListener('click', () => onSelect(id))
    districts.push({ id, root, label, bounds, outline })
    return root
  }
  function slots(parent: THREE.Object3D, origins: THREE.Vector3[], columns: number, spacing: number, height: number) {
    const mesh = new THREE.InstancedMesh(cube, material('#ffffff'), origins.length * 48)
    const transform = new THREE.Object3D()
    origins.forEach((origin, groupIndex) => {
      for (let index = 0; index < 48; index += 1) {
        transform.position.set(origin.x + (index % columns - (columns - 1) / 2) * spacing, height, origin.z + (Math.floor(index / columns) - (48 / columns - 1) / 2) * spacing)
        transform.scale.set(spacing * 0.66, 0.09, spacing * 0.66); transform.updateMatrix()
        mesh.setMatrixAt(groupIndex * 48 + index, transform.matrix)
        mesh.setColorAt(groupIndex * 48 + index, new THREE.Color('#566453'))
      }
    })
    mesh.instanceMatrix.needsUpdate = true; parent.add(mesh); warpMeshes.push(mesh)
  }
  function connect(points: [number, number, number][], color: string, engine: Flow['engine']) {
    const curve = new THREE.CurvePath<THREE.Vector3>()
    for (let index = 1; index < points.length; index += 1) {
      const start = new THREE.Vector3(...points[index - 1]); const end = new THREE.Vector3(...points[index])
      if (start.distanceTo(end) > 0.001) curve.add(new THREE.LineCurve3(start, end))
    }
    const geometry = new THREE.TubeGeometry(curve, 32, 0.045, 5, false); transient.push(geometry)
    world.add(new THREE.Mesh(geometry, material(color)))
    const arrowGeometry = new THREE.ConeGeometry(0.14, 0.34, 5); transient.push(arrowGeometry)
    const arrow = new THREE.Mesh(arrowGeometry, material(color))
    arrow.position.copy(curve.getPoint(0.76)); arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(0.76)); world.add(arrow)
    const packets = Array.from({ length: 6 }, () => box(world, [0.22, 0.14, 0.22], [0, 0, 0], color))
    flows.push({ curve, packets, engine })
  }
  function clear() {
    for (const district of districts) {
      district.label.remove(); district.outline.geometry.dispose(); (district.outline.material as THREE.Material).dispose()
    }
    for (const geometry of transient) geometry.dispose()
    for (const mesh of warpMeshes) mesh.dispose()
    districts.length = 0; flows.length = 0; transient.length = 0; warpMeshes.length = 0; world.clear()
  }
  function buildGpu() {
    box(world, [33, 0.5, 23], [-1.5, -0.5, 1], '#293333')
    box(world, [21, 0.2, 14], [1.2, -0.08, -1.4], '#47504a')
    box(world, [20.5, 0.08, 13.5], [1.2, 0.06, -1.4], '#202c25')
    for (const side of [-1, 1]) for (let pin = 0; pin < 24; pin += 1) box(world, [0.22, 0.07, 0.55], [pin * 0.72 - 7.2, 0, side * 7 - 1.4], '#b0a373')
    node('host', [-14, 0, -2], (group) => {
      box(group, [3.5, 0.22, 3.5], [0, 0.1, 0], '#6b7b80')
      box(group, [2.7, 0.65, 2.7], [0, 0.5, 0], '#5b7d96')
      for (let index = 0; index < 5; index += 1) box(group, [0.26, 0.5, 2.5], [(index - 2) * 0.48, 1.08, 0], '#9bb8ce')
    })
    node('dispatch', [-8, 0, -2], (group) => { bank(group, [0, 0, 0], '#cbb568', 3) })
    node('copy', [-13, 0, 6.5], (group) => { bank(group, [0, 0, 0], '#76b9bc', 4) })
    node('sm', [1.8, 0, -2.2], (group) => {
      const origins: THREE.Vector3[] = []
      for (let row = 0; row < 2; row += 1) for (let column = 0; column < 4; column += 1) {
        const position = new THREE.Vector3((column - 1.5) * 3.3, 0, (row - 0.5) * 4.0)
        origins.push(position)
        box(group, [2.8, 0.4, 3.2], [position.x, 0.35, position.z], '#566c4d')
        box(group, [2.45, 1, 2.65], [position.x, 1.02, position.z], '#668d46')
        box(group, [2.5, 0.18, 0.4], [position.x, 0.75, position.z + 1.35], '#77cbc4')
        for (let unit = 0; unit < 4; unit += 1) box(group, [0.34, 0.4, 0.4], [position.x + (unit - 1.5) * 0.51, 1.7, position.z - 0.95], '#d4bd6b')
      }
      slots(group, origins, 8, 0.25, 1.58)
    })
    node('l2', [1, 0, 4.7], (group) => { bank(group, [0, 0, 0], '#63a8c3', 24) })
    node('dram', [1, 0, 9.2], (group) => {
      for (let index = 0; index < 6; index += 1) {
        box(group, [2, 0.25, 1.9], [(index - 2.5) * 2.5, 0.18, 0], '#77898d')
        box(group, [1.65, 0.75, 1.5], [(index - 2.5) * 2.5, 0.65, 0], '#80bacc')
      }
    })
    connect([[-12, 0.55, -2], [-9, 0.55, -2]], '#edcf7c', 'command')
    connect([[-7, 0.55, -2], [-5.5, 0.55, -2], [-5.5, 0.55, -4.2], [-4.5, 0.55, -4.2]], '#edcf7c', 'command')
    connect([[-14, 0.5, 0], [-14, 0.5, 5.3], [-13, 0.5, 5.3]], '#71d0df', 'memory')
    connect([[-11.5, 0.5, 6.5], [-10, 0.5, 6.5], [-10, 0.5, 9.2], [-6.8, 0.5, 9.2]], '#71d0df', 'memory')
    for (let column = 0; column < 4; column += 1) {
      const horizontal = 1.8 + (column - 1.5) * 3.3
      connect([[horizontal, 0.52, 8.3], [horizontal, 0.52, 5.4]], '#71d0df', 'memory')
      connect([[horizontal, 0.52, 4], [horizontal, 0.52, 1.6]], '#71d0df', 'memory')
      connect([[horizontal + 0.4, 0.6, 1.6], [horizontal + 0.4, 0.6, 4]], '#dda774', 'memory')
    }
  }
  function buildSm() {
    box(world, [25, 0.5, 18], [0, -0.5, 0.5], '#293333')
    box(world, [24, 0.12, 17], [0, -0.15, 0.5], '#4b574a')
    node('scheduler', [0, 0, -5.6], (group) => {
      box(group, [14.5, 0.6, 2.5], [0, 0.2, 0], '#7d7c5f')
      slots(group, [new THREE.Vector3()], 12, 0.48, 0.58)
    })
    node('registers', [0, 0, -0.8], (group) => {
      for (let row = 0; row < 3; row += 1) bank(group, [0, row * 0.45, row * 0.2], '#a6bcc4', 6)
    })
    node('shared', [-1.5, 0, 5.6], (group) => { bank(group, [0, 0, 0], '#72c6bc', 17) })
    node('cuda', [-7.1, 0, -0.4], (group) => {
      box(group, [4.6, 0.22, 5.5], [0, 0.08, 0], '#657950')
      for (let row = 0; row < 5; row += 1) for (let column = 0; column < 4; column += 1) box(group, [0.6, 1.1, 0.7], [(column - 1.5) * 0.92, 0.76, (row - 2) * 0.93], '#a1ce64')
    })
    node('tensor', [6.5, 0, -0.4], (group) => {
      box(group, [4.6, 0.22, 5.5], [0, 0.08, 0], '#857545')
      for (let layer = 0; layer < 3; layer += 1) for (let index = 0; index < 4; index += 1) box(group, [0.55, 0.32, 3.8], [(index - 1.5) * 0.85, 0.4 + layer * 0.6, 0], '#e7c56c')
    })
    node('rt', [8, 0, 5.6], (group) => {
      box(group, [3.8, 0.22, 3], [0, 0.08, 0], '#6b565e')
      for (let index = 0; index < 3; index += 1) {
        const geometry = new THREE.OctahedronGeometry(0.52); transient.push(geometry)
        const mesh = new THREE.Mesh(geometry, material('#987082'))
        mesh.position.set((index - 1) * 0.95, 1.05, index === 1 ? -0.45 : 0.35); group.add(mesh)
      }
    })
    connect([[-4.7, 0.65, -5], [-7, 0.65, -5], [-7, 0.65, -3.3]], '#edcf7c', 'command')
    connect([[4.7, 0.65, -5], [6.5, 0.65, -5], [6.5, 0.65, -3.3]], '#edcf7c', 'command')
    connect([[0, 0.52, 4.8], [0, 0.52, 1.5]], '#71d0df', 'memory')
    connect([[-1.8, 0.65, -0.8], [-4.7, 0.65, -0.8]], '#a8dc6b', 'cuda')
    connect([[1.8, 0.65, -0.8], [4.1, 0.65, -0.8]], '#e8c26f', 'tensor')
    connect([[-6, 0.5, 2.6], [-6, 0.5, 4], [-2.5, 0.5, 4], [-2.5, 0.5, 4.9]], '#dda774', 'cuda')
    connect([[6.5, 0.5, 2.6], [6.5, 0.5, 3.8], [2, 0.5, 3.8], [2, 0.5, 4.9]], '#dda774', 'tensor')
  }
  function select(id: DistrictId) {
    selected = id
    for (const district of districts) {
      district.outline.visible = district.id === id
      district.label.classList.toggle('selected', district.id === id)
      district.label.setAttribute('aria-pressed', String(district.id === id))
    }
  }
  function move(target: THREE.Vector3, position: THREE.Vector3, immediate = false) {
    if (reduced.matches || immediate) { controls.target.copy(target); camera.position.copy(position); flight = null }
    else flight = { target, position }
  }
  function home(immediate = false) {
    const halfSpan = view === 'gpu' ? 20 : 15
    const distance = Math.max(view === 'gpu' ? 40 : 31, halfSpan / (Math.tan(THREE.MathUtils.degToRad(21)) * Math.max(0.35, camera.aspect)))
    const target = new THREE.Vector3(view === 'gpu' ? -1.5 : 0, 0, view === 'gpu' ? 1 : 0)
    move(target, target.clone().add(new THREE.Vector3(0.32, 0.95, 1.12).normalize().multiplyScalar(distance)), immediate)
  }
  function setView(next: View) {
    view = next; clear()
    if (view === 'gpu') buildGpu(); else buildSm()
    select(selected); home(true)
    host.dataset.view = view; host.dataset.districts = String(districts.length)
  }
  function focus(id: DistrictId) {
    const district = districts.find((item) => item.id === id)
    if (!district) return
    select(id)
    const target = district.bounds.getCenter(new THREE.Vector3())
    const size = district.bounds.getSize(new THREE.Vector3()).length()
    const offset = new THREE.Vector3(0.2, 0.8, 1).normalize().multiplyScalar(Math.max(13, size * 1.6))
    move(target, target.clone().add(offset))
  }
  const raycaster = new THREE.Raycaster()
  let downX = 0; let downY = 0
  function hit(event: PointerEvent) {
    const rect = renderer.domElement.getBoundingClientRect()
    raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, 1 - (event.clientY - rect.top) / rect.height * 2), camera)
    let object: THREE.Object3D | null = raycaster.intersectObjects(districts.map((district) => district.root), true)[0]?.object ?? null
    while (object) {
      if (object.userData.districtId) return object.userData.districtId as DistrictId
      object = object.parent
    }
    return null
  }
  renderer.domElement.addEventListener('pointerdown', (event) => { downX = event.clientX; downY = event.clientY; flight = null })
  renderer.domElement.addEventListener('pointerup', (event) => {
    if (Math.hypot(event.clientX - downX, event.clientY - downY) > 5) return
    const id = hit(event); if (id) onSelect(id)
  })
  renderer.domElement.addEventListener('pointermove', (event) => { renderer.domElement.style.cursor = hit(event) ? 'pointer' : 'grab' })
  controls.addEventListener('start', () => { flight = null })
  const observer = new ResizeObserver(() => {
    const width = Math.max(1, host.clientWidth); const height = Math.max(1, host.clientHeight)
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); home(true)
  })
  observer.observe(host)

  function render(elapsed: number, sample: Sample, kernel: Kernel, precision: Precision, delta: number) {
    if (flight) {
      const amount = 1 - Math.exp(-delta * 7)
      camera.position.lerp(flight.position, amount); controls.target.lerp(flight.target, amount)
      if (camera.position.distanceTo(flight.position) < 0.02) flight = null
    }
    const path = executionPath(kernel, precision)
    const color = new THREE.Color()
    for (const mesh of warpMeshes) {
      for (let index = 0; index < mesh.count; index += 1) {
        const occupied = index % 48 < sample.activeWarps
        color.set(occupied ? '#d4efa6' : '#394737')
        if (occupied) color.multiplyScalar(0.85 + 0.15 * Math.sin(elapsed * 4 + index * 0.5))
        mesh.setColorAt(index, color)
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
    for (const flow of flows) {
      const enabled = sample.activeWarps > 0 && (flow.engine === 'memory' || flow.engine === 'command' || flow.engine === path.engine)
      flow.packets.forEach((packet, index) => {
        packet.visible = enabled
        const progress = (elapsed * 0.34 + index / flow.packets.length) % 1
        packet.position.copy(flow.curve.getPoint(progress)); packet.position.y += 0.12
      })
    }
    controls.update(); renderer.render(scene, camera)
    const occupied: DOMRect[] = []
    let framed = 0
    const sorted = [...districts].sort((first, second) => Number(second.id === selected) - Number(first.id === selected))
    for (const district of sorted) {
      const point = district.bounds.getCenter(new THREE.Vector3()); point.y = district.bounds.max.y + 0.65; point.project(camera)
      const left = (point.x * 0.5 + 0.5) * host.clientWidth
      const top = (0.5 - point.y * 0.5) * host.clientHeight
      const labelWidth = host.clientWidth < 650 ? 108 : 150
      const labelHeight = host.clientWidth < 650 ? 28 : 44
      const rect = new DOMRect(left - labelWidth / 2 - 4, top - labelHeight, labelWidth + 8, labelHeight + 5)
      const overlaps = occupied.some((other) => rect.left < other.right && rect.right > other.left && rect.top < other.bottom && rect.bottom > other.top)
      district.label.hidden = !showLabels || Math.abs(point.z) > 1 || rect.left < 8 || rect.right > host.clientWidth - 8 || rect.top < 75 || rect.bottom > host.clientHeight - 52 || overlaps
      if (!district.label.hidden) { occupied.push(rect); district.label.style.left = `${left}px`; district.label.style.top = `${top}px` }
      let inside = true
      for (const horizontal of [district.bounds.min.x, district.bounds.max.x]) for (const vertical of [district.bounds.min.y, district.bounds.max.y]) for (const depth of [district.bounds.min.z, district.bounds.max.z]) {
        const corner = new THREE.Vector3(horizontal, vertical, depth).project(camera)
        if (Math.abs(corner.x) > 0.98 || Math.abs(corner.y) > 0.98 || Math.abs(corner.z) > 1) inside = false
      }
      if (inside) framed += 1
    }
    host.dataset.rendered = 'true'; host.dataset.framed = String(framed)
    host.dataset.activeWarps = String(sample.activeWarps); host.dataset.engine = path.engine
  }
  setView('gpu')
  return {
    setView, home, select, focus, render,
    setLabels(visible: boolean) { showLabels = visible },
    setDay(day: boolean) { scene.background = new THREE.Color(day ? '#e2e8e4' : '#151819'); ambient.intensity = day ? 3.4 : 2.5 },
    zoom(direction: number) {
      const target = controls.target.clone(); const offset = camera.position.clone().sub(target)
      offset.setLength(THREE.MathUtils.clamp(offset.length() * (direction > 0 ? 0.8 : 1.25), controls.minDistance, controls.maxDistance))
      move(target, target.clone().add(offset))
    },
    dispose() { observer.disconnect(); controls.dispose(); clear(); cube.dispose(); for (const entry of materials.values()) entry.dispose(); renderer.dispose(); host.replaceChildren() },
  }
}