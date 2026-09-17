import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import './style.css'
import { calculateOccupancy, sampleKernel, type Kernel } from './sim/model'

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
  <header><a class="brand" href="./"><b>CUDA</b> SimCity</a><div class="live"><i></i><span>Kernel running</span></div><nav><button id="pause">Pause</button><button id="reset">Reset</button><a href="https://github.com/eoinjordan/CUDASimCity">Source</a></nav></header>
  <main><section class="viewport"><div id="scene"></div><div class="title"><p>Parallel computing, made visible</p><h1>Schedule the whole city.</h1></div><div class="key"><span><i class="compute"></i>Compute</span><span><i class="memory"></i>Memory</span><span><i class="warp"></i>Warp</span></div></section>
  <aside><section><p class="eyebrow">Kernel</p><div class="modes"><button class="active" data-kernel="matrix">Matrix multiply</button><button data-kernel="reduction">Reduction</button><button data-kernel="stencil">Stencil</button></div><label>Threads per block <output id="threads-value">256</output><input id="threads" type="range" min="64" max="1024" step="64" value="256"></label><label>Registers per thread <output id="registers-value">32</output><input id="registers" type="range" min="16" max="128" step="8" value="32"></label><label>Shared memory <output id="shared-value">12 KB</output><input id="shared" type="range" min="0" max="96" step="4" value="12"></label></section>
  <section class="metrics"><p class="eyebrow">Scheduler state</p><div><strong id="occupancy">0%</strong><span>occupancy</span></div><div><strong id="warps">0</strong><span>active warps</span></div><div><strong id="compute">0%</strong><span>compute use</span></div><div><strong id="memory">0%</strong><span>memory flow</span></div><div><strong id="divergence">0%</strong><span>divergence</span></div></section>
  <section class="detail"><p class="eyebrow">City map</p><h2>Streaming multiprocessors</h2><p>Each tower is an illustrative SM. Warp lights travel between the scheduler grid, shared memory, and global memory.</p></section><footer>Simplified educational model. Values are not measurements or specifications for a particular GPU.</footer></aside></main>`

const host = document.querySelector<HTMLDivElement>('#scene')!
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101613); scene.fog = new THREE.Fog(0x101613, 18, 36)
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100); camera.position.set(12, 11, 15)
const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.outputColorSpace = THREE.SRGBColorSpace; host.append(renderer.domElement)
const controls = new OrbitControls(camera, renderer.domElement); controls.target.set(0, .7, 0); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * .48; controls.minDistance = 9; controls.maxDistance = 29
scene.add(new THREE.HemisphereLight(0xd9ffe1, 0x101613, 2.2)); const sun = new THREE.DirectionalLight(0xcaffbd, 3.7); sun.position.set(-8, 14, 8); sun.castShadow = true; scene.add(sun)
const base = new THREE.Mesh(new THREE.CylinderGeometry(8.8, 9.2, .5, 8), new THREE.MeshStandardMaterial({ color: 0x1d2923, roughness: .85 })); base.position.y = -.35; base.receiveShadow = true; scene.add(base)

const sms: THREE.Mesh[] = []
for (let row = 0; row < 3; row += 1) for (let column = 0; column < 5; column += 1) {
  const height = 1.7 + ((row * 5 + column) % 3) * .38
  const sm = new THREE.Mesh(new THREE.BoxGeometry(1.15, height, 1.15), new THREE.MeshStandardMaterial({ color: 0x53c76f, emissive: 0x102d18, roughness: .48, metalness: .22 }))
  sm.position.set((column - 2) * 1.55, height / 2, (row - 1) * 1.55); sm.castShadow = true; sm.receiveShadow = true; scene.add(sm); sms.push(sm)
}
const memory = new THREE.Mesh(new THREE.BoxGeometry(7.8, 1.1, 1.25), new THREE.MeshStandardMaterial({ color: 0x3d6ecc, emissive: 0x0e1e45, roughness: .42, metalness: .28 })); memory.position.set(0, .55, 4); memory.castShadow = true; scene.add(memory)
const warpGeometry = new THREE.SphereGeometry(.11, 12, 12)
const warpLights = Array.from({ length: 48 }, (_, index) => { const light = new THREE.Mesh(warpGeometry, new THREE.MeshBasicMaterial({ color: index % 4 === 0 ? 0x75a8ff : 0xc6ff68 })); light.userData.phase = index / 48; scene.add(light); return light })

let kernel: Kernel = 'matrix'; let running = true; let tick = 0; let elapsed = 0
const threads = document.querySelector<HTMLInputElement>('#threads')!; const registers = document.querySelector<HTMLInputElement>('#registers')!; const shared = document.querySelector<HTMLInputElement>('#shared')!
function updateModel(): number {
  const occupancy = calculateOccupancy(Number(threads.value), Number(registers.value), Number(shared.value)); const sample = sampleKernel(kernel, occupancy, tick++)
  document.querySelector('#occupancy')!.textContent = `${Math.round(occupancy * 100)}%`; document.querySelector('#warps')!.textContent = String(sample.activeWarps); document.querySelector('#compute')!.textContent = `${Math.round(sample.computeUse * 100)}%`; document.querySelector('#memory')!.textContent = `${Math.round(sample.memoryUse * 100)}%`; document.querySelector('#divergence')!.textContent = `${Math.round(sample.divergence * 100)}%`
  warpLights.forEach((light, index) => { light.visible = index < sample.activeWarps * .75 }); sms.forEach((sm, index) => { (sm.material as THREE.MeshStandardMaterial).emissiveIntensity = .25 + sample.computeUse * .8 * (index % 3 === tick % 3 ? 1 : .5) }); return occupancy
}
function reset(): void { tick = 0; elapsed = 0; updateModel() }
document.querySelector('#pause')!.addEventListener('click', (event) => { running = !running; (event.currentTarget as HTMLButtonElement).textContent = running ? 'Pause' : 'Resume'; document.querySelector('.live i')!.classList.toggle('paused', !running); document.querySelector('.live span')!.textContent = running ? 'Kernel running' : 'Kernel paused' })
document.querySelector('#reset')!.addEventListener('click', reset)
document.querySelectorAll<HTMLButtonElement>('[data-kernel]').forEach((button) => button.addEventListener('click', () => { kernel = button.dataset.kernel as Kernel; document.querySelectorAll('[data-kernel]').forEach((item) => item.classList.toggle('active', item === button)); reset() }))
for (const input of [threads, registers, shared]) input.addEventListener('input', () => { document.querySelector(`#${input.id}-value`)!.textContent = `${input.value}${input === shared ? ' KB' : ''}`; updateModel() })

const clock = new THREE.Clock()
function resize(): void { renderer.setSize(host.clientWidth, host.clientHeight, false); camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix() }
function animate(): void { requestAnimationFrame(animate); const delta = Math.min(clock.getDelta(), .1); if (running && !matchMedia('(prefers-reduced-motion: reduce)').matches) { elapsed += delta; if (Math.floor(elapsed * 3) > tick) updateModel(); warpLights.forEach((light) => { const progress = (elapsed * .18 + light.userData.phase) % 1; const target = sms[Math.floor(light.userData.phase * sms.length)]; light.position.lerpVectors(new THREE.Vector3(0, 1.2, 4), new THREE.Vector3(target.position.x, target.position.y + 1.2, target.position.z), progress); light.position.y += Math.sin(progress * Math.PI) * 2.2 }) } controls.update(); renderer.render(scene, camera) }
new ResizeObserver(resize).observe(host); resize(); reset(); animate()