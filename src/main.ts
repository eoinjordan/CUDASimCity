import './style.css'
import { Timer } from 'three'
import { createElement, Cpu, Pause, Play, RotateCcw, Focus, ZoomIn, ZoomOut, Sun, Moon, Route, ChevronLeft, ChevronRight, X, Download, Copy, SkipForward, type IconNode } from 'lucide'
import { ADA_SM, PRECISIONS, calculateResidency, executionPath, quantizationExample, sampleKernel, storageBytes, type Kernel, type Precision } from './sim/model'
import { DISTRICTS, SOURCES, TOUR, type DistrictId, type View } from './sim/architecture'
import { createCity } from './world/city'

const icon = (node: IconNode) => createElement(node, { width: '18', height: '18', 'aria-hidden': 'true', 'stroke-width': '1.7' }).outerHTML
const tool = (id: string, title: string, node: IconNode) => `<button id="${id}" class="icon-button" aria-label="${title}" title="${title}">${icon(node)}</button>`
const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
  <header class="app-header"><a class="brand" href="./">${icon(Cpu)}<span>CUDA <strong>SimCity</strong></span></a><div class="view-switch segmented" role="group" aria-label="Architecture view"><button data-view="gpu" aria-pressed="true">GPU die</button><button data-view="sm" aria-pressed="false">Inside an SM</button></div><div class="header-tools"><button id="tour-start" class="command">${icon(Route)}<span>Guided tour</span></button>${tool('theme', 'Switch to day view', Sun)}${tool('export', 'Export simulation snapshot', Download)}</div></header>
  <main class="workspace">
    <section class="stage" aria-label="CUDA architecture workspace">
      <div class="metrics"><div><span>OCCUPANCY BOUND</span><strong id="occupancy">100%</strong></div><div><span>BLOCK CAPACITY / SM</span><strong id="blocks">6</strong></div><div><span>WARP CAPACITY / SM</span><strong id="warps">48<small>/ 48</small></strong></div><div class="execution-metric"><span>EXAMPLE EXECUTION</span><strong id="engine">TENSOR MMA</strong></div></div>
      <div class="viewport"><div id="scene"></div><div class="scene-heading"><p class="eyebrow">NVIDIA / ADA REFERENCE</p><h1 id="scene-title">GPU architecture</h1><p id="scene-caption">Host, blocks, caches, and resident warps</p></div><div class="camera-tools">${tool('home', 'Frame the architecture', Focus)}${tool('zoom-in', 'Zoom in', ZoomIn)}${tool('zoom-out', 'Zoom out', ZoomOut)}<label><input id="labels" type="checkbox" checked>Labels</label></div><div class="legend"><span><i style="--swatch:#edcf7c"></i>Commands</span><span><i style="--swatch:#71d0df"></i>Data loads</span><span><i style="--swatch:#dda774"></i>Results</span><span><i style="--swatch:#cbe9a0"></i>Resident warps</span></div><section id="tour" class="tour-panel" hidden aria-label="CUDA guided tour"><div class="tour-top"><p id="tour-count" class="eyebrow"></p>${tool('tour-close', 'End tour', X)}</div><h2 id="tour-title"></h2><p id="tour-body"></p><div class="tour-bottom">${tool('tour-prev', 'Previous stage', ChevronLeft)}<div id="tour-progress"></div>${tool('tour-next', 'Next stage', ChevronRight)}</div></section></div>
      <div class="sm-strip"><div class="resident-panel"><div class="section-title"><span class="eyebrow">RESIDENT WARPS / MODEL SM</span><span id="resident-count">48 / 48</span></div><div id="warp-slots" class="warp-slots" aria-label="48 model warp residency slots">${Array.from({ length: 48 }, (_, index) => `<i title="Warp slot ${index + 1}"></i>`).join('')}</div><p id="limiter">Capacity limited by warp slots</p></div><div class="storage-panel"><span class="eyebrow">TWO 64 x 64 INPUT TILES</span><strong id="tile-storage">16<small>KiB</small></strong><span id="accumulator">FP32 accumulation</span></div></div>
      <div class="transport"><div>${tool('pause', 'Pause simulation', Pause)}${tool('step', 'Step simulation', SkipForward)}${tool('reset', 'Reset reference defaults', RotateCcw)}<span id="clock">0.0s</span></div><span id="run-state">RUNNING</span><span class="disclaimer">ILLUSTRATIVE ACTIVITY / NOT GPU TELEMETRY</span></div>
    </section>
    <aside class="sidebar"><div class="reference-header"><p class="eyebrow">COMPUTE CAPABILITY 8.9</p><h2>Ada SM reference</h2><p>Per-SM capacity model. No GPU connected.</p></div><div class="tabs" role="tablist" aria-label="CUDA detail panels"><button role="tab" id="tab-kernel" data-tab="kernel" aria-selected="true" aria-controls="panel-kernel">Kernel</button><button role="tab" id="tab-inspect" data-tab="inspect" aria-selected="false" aria-controls="panel-inspect" tabindex="-1">Inspect</button><button role="tab" id="tab-precision" data-tab="precision" aria-selected="false" aria-controls="panel-precision" tabindex="-1">Precision</button><button role="tab" id="tab-toolkit" data-tab="toolkit" aria-selected="false" aria-controls="panel-toolkit" tabindex="-1">Toolkit</button></div><div class="panel-body">
      <div id="panel-kernel" role="tabpanel" aria-labelledby="tab-kernel"><section class="panel-section"><h3>Workload</h3><div class="workloads"><button data-kernel="matrix" aria-pressed="true">Matrix multiply</button><button data-kernel="reduction" aria-pressed="false">Reduction</button><button data-kernel="stencil" aria-pressed="false">Stencil</button><button data-kernel="idle" aria-pressed="false">Idle</button></div><h3 class="subheading">Input / math format</h3><div class="formats segmented" role="group" aria-label="Precision">${Object.entries(PRECISIONS).map(([id, format]) => `<button data-precision="${id}" aria-pressed="${id === 'fp16'}">${format.name}</button>`).join('')}</div><p id="path-description" class="fine-print"></p></section>
      <section class="panel-section"><h3>Per-block resources</h3><label for="threads" class="slider-field"><span>Threads per block<output id="threads-value">256</output></span><input id="threads" type="range" min="32" max="1024" step="32" value="256"></label><label for="registers" class="slider-field"><span>32-bit registers per thread<output id="registers-value">32</output></span><input id="registers" type="range" min="16" max="128" step="8" value="32"></label><label for="shared" class="slider-field"><span>Shared memory per block<output id="shared-value">12 KiB</output></span><input id="shared" type="range" min="0" max="96" step="4" value="12"></label><p class="fine-print">100 KiB shared carveout; 1 KiB reserved per block. Allocations above 48 KiB need dynamic shared-memory opt-in.</p></section>
      <section class="panel-section"><h3>Capacity constraints</h3><dl id="resource-limits" class="resource-limits"></dl><p class="fine-print">A whole-block capacity bound, not the occupancy API. Register allocation granularity and SM partitions can lower actual residency.</p></section><section class="panel-section"><h3>Illustrative activity</h3><div class="activity-row"><span>CUDA arithmetic</span><meter id="cuda-activity" min="0" max="1"></meter></div><div class="activity-row"><span>Tensor MMA</span><meter id="tensor-activity" min="0" max="1"></meter></div><div class="activity-row"><span>Memory requests</span><meter id="memory-activity" min="0" max="1"></meter></div><div class="activity-row"><span>RT Core</span><strong>Inactive</strong></div></section></div>
      <div id="panel-inspect" role="tabpanel" aria-labelledby="tab-inspect" hidden><section class="panel-section" id="inspector"></section><section class="panel-section"><h3>Architecture index</h3><div id="district-index"></div></section></div>
      <div id="panel-precision" role="tabpanel" aria-labelledby="tab-precision" hidden><section class="panel-section"><p class="eyebrow">VERIFIED STORAGE ARITHMETIC</p><h2>Precision is not a speed multiplier.</h2><p class="description">TF32 changes matrix arithmetic while retaining FP32 storage. FP16 takes two bytes per stored element; INT8 takes one, before scales, padding, and other metadata.</p><dl class="facts"><div><dt>Example operation</dt><dd id="precision-operation"></dd></div><div><dt>Input payload</dt><dd id="precision-payload"></dd></div></dl></section><section class="panel-section"><h3>INT8 arithmetic reference</h3><p class="fine-print">Zero point 0, nearest-even rounding, and clipping to [-128, 127]. The example scale is max(abs(x)) / 127; it is not model calibration.</p><div id="quantization"></div></section></div>
      <div id="panel-toolkit" role="tabpanel" aria-labelledby="tab-toolkit" hidden><section class="panel-section"><h3>Software, not silicon</h3><dl class="facts"><div><dt>CUDA Toolkit</dt><dd>Compiler, runtime libraries, accelerated libraries, and development tools.</dd></div><div><dt>CUDA programming model</dt><dd>Host launch, grids, blocks, threads, memory, and synchronization.</dd></div><div><dt>RTX hardware platform</dt><dd>Programmable execution, specialized RT resources, and Tensor resources for supported rendering/AI workloads.</dd></div></dl></section><section class="panel-section"><div class="section-title"><h3>Target verification</h3>${tool('copy-commands', 'Copy verification commands', Copy)}</div><pre><code id="native-commands">nvcc --version
nvidia-smi
ncu --set full ./your-cuda-app</code></pre><p class="fine-print">Run on a supported NVIDIA target with the required tools. This browser app runs on macOS too, but does not use Metal or CPU timings as CUDA measurements.</p></section><section class="panel-section source-links"><h3>Primary references</h3><a href="${SOURCES.rtx}" target="_blank" rel="noreferrer">NVIDIA RTX</a><a href="${SOURCES.toolkit}" target="_blank" rel="noreferrer">CUDA Toolkit</a><a href="${SOURCES.model}" target="_blank" rel="noreferrer">CUDA programming model</a><a href="${SOURCES.ada}" target="_blank" rel="noreferrer">Ada tuning guide / capacity limits</a><a href="${SOURCES.occupancy}" target="_blank" rel="noreferrer">Occupancy and performance</a></section></div>
    </div><footer><span>Independent educational model</span><a href="https://github.com/eoinjordan/CUDASimCity" target="_blank" rel="noreferrer">Source</a></footer></aside>
  </main><div id="notice" role="status" aria-live="polite"></div>`

const get = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const reduced = matchMedia('(prefers-reduced-motion: reduce)')
let running = !reduced.matches
let elapsed = 0
let kernel: Kernel = 'matrix'
let precision: Precision = 'fp16'
let view: View = 'gpu'
let selected: DistrictId = 'sm'
let tab = 'kernel'
let day = false
let tourIndex: number | null = null
let tourTime = 0
let city: ReturnType<typeof createCity> | null = null
let noticeTimer: ReturnType<typeof setTimeout> | undefined
const inputs = { threads: get<HTMLInputElement>('threads'), registers: get<HTMLInputElement>('registers'), shared: get<HTMLInputElement>('shared') }
let residency = calculateResidency(256, 32, 12)
let sample = sampleKernel(kernel, residency.occupancy, 0)

function setTab(next: string) {
  tab = next
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-tab]')) {
    const active = button.dataset.tab === tab
    button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1
    get(`panel-${button.dataset.tab}`).hidden = !active
  }
}

function switchView(next: View) {
  view = next; city?.setView(view)
  get('scene-title').textContent = view === 'gpu' ? 'GPU architecture' : 'Inside a streaming multiprocessor'
  get('scene-caption').textContent = view === 'gpu' ? 'Eight representative SMs / not a GPU SKU' : 'Resident warps, local state, and functional units'
  for (const button of document.querySelectorAll('[data-view]')) button.setAttribute('aria-pressed', String((button as HTMLElement).dataset.view === view))
  if (DISTRICTS.find((district) => district.id === selected)!.view !== view) selected = view === 'gpu' ? 'sm' : 'scheduler'
  inspect(selected)
}

function inspect(id: DistrictId, focus = false) {
  const district = DISTRICTS.find((entry) => entry.id === id)!
  if (district.view !== view) switchView(district.view)
  selected = id; city?.select(id)
  if (focus) city?.focus(id)
  get('inspector').innerHTML = `<p class="eyebrow" style="color:${district.color}">${district.view === 'gpu' ? 'GPU / SYSTEM CONTEXT' : 'SM / EXECUTION CONTEXT'}</p><h2>${district.name}</h2><p class="subtitle">${district.subtitle}</p><p class="description">${district.description}</p><dl class="facts">${district.facts.map(([name, value]) => `<div><dt>${name}</dt><dd>${value}</dd></div>`).join('')}</dl><div class="caution"><span>WHAT THIS DOES NOT CLAIM</span><p>${district.boundary}</p></div><a href="${district.source}" target="_blank" rel="noreferrer" class="reference-link">Official reference ${icon(ChevronRight)}</a>`
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-inspect]')) button.setAttribute('aria-pressed', String(button.dataset.inspect === id))
}

try { city = createCity(get<HTMLDivElement>('scene'), (id) => { stopTour(); inspect(id); setTab('inspect') }) }
catch (error) { get('scene').innerHTML = '<div class="scene-error" role="alert"><h2>WebGL2 unavailable</h2><p>The model, precision lab, and inspector remain available.</p></div>'; console.warn('WebGL initialization failed', error) }

get('district-index').innerHTML = DISTRICTS.map((district) => `<button data-inspect="${district.id}" aria-pressed="false"><i style="--swatch:${district.color}"></i><span><strong>${district.name}</strong><small>${district.subtitle}</small></span>${icon(ChevronRight)}</button>`).join('')
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-inspect]')) button.addEventListener('click', () => { stopTour(); inspect(button.dataset.inspect as DistrictId, true) })

function updateModel() {
  residency = calculateResidency(Number(inputs.threads.value), Number(inputs.registers.value), Number(inputs.shared.value))
  sample = sampleKernel(kernel, residency.occupancy, Math.floor(elapsed * 3))
  const path = executionPath(kernel, precision)
  get('occupancy').textContent = `${Math.round(residency.occupancy * 100)}%`
  get('blocks').textContent = String(residency.blocksPerSm)
  get('warps').innerHTML = `${residency.residentWarps}<small>/ ${ADA_SM.maxWarps}</small>`
  get('engine').textContent = sample.activeWarps === 0 ? 'NO ACTIVE WORK' : path.engine === 'tensor' ? 'TENSOR MMA' : 'CUDA / SIMT'
  get('path-description').textContent = kernel === 'idle' ? 'No kernel is dispatched. Capacity bounds remain visible; resident work is zero.' : `${path.instruction}. ${PRECISIONS[precision].description}`
  get('resident-count').textContent = `${sample.activeWarps} / ${ADA_SM.maxWarps}`
  for (const [index, slot] of [...get('warp-slots').children].entries()) slot.classList.toggle('occupied', index < sample.activeWarps)
  get('limiter').textContent = residency.blocksPerSm === 0 ? 'No whole block fits these resource settings.' : `Capacity bound: ${residency.limitingResources.join(' + ')}`
  get('limiter').classList.toggle('warning', residency.blocksPerSm === 0)
  get('resource-limits').innerHTML = Object.entries(residency.limits).map(([name, blocks]) => `<div class="${blocks === residency.blocksPerSm ? 'limiting' : ''}"><dt>${name}</dt><dd>${blocks} blocks</dd></div>`).join('')
  const bytes = storageBytes(2 * 64 * 64, precision)
  get('tile-storage').innerHTML = `${bytes / 1024}<small>KiB</small>`
  get('accumulator').textContent = `${PRECISIONS[precision].accumulator} matrix accumulator`
  get('precision-operation').textContent = path.instruction
  get('precision-payload').textContent = `${bytes.toLocaleString()} bytes / excludes scales and padding`
  get<HTMLMeterElement>('cuda-activity').value = path.engine === 'cuda' ? sample.computeUse : sample.computeUse * 0.2
  get<HTMLMeterElement>('tensor-activity').value = path.engine === 'tensor' ? sample.computeUse : 0
  get<HTMLMeterElement>('memory-activity').value = sample.memoryUse
  for (const [name, input] of Object.entries(inputs)) get(`${name}-value`).textContent = `${input.value}${name === 'shared' ? ' KiB' : ''}`
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-kernel]')) button.setAttribute('aria-pressed', String(button.dataset.kernel === kernel))
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-precision]')) {
    button.setAttribute('aria-pressed', String(button.dataset.precision === precision))
    button.disabled = button.dataset.precision === 'tf32' && kernel !== 'matrix'
  }
  get('clock').textContent = `${elapsed.toFixed(1)}s`
  get('run-state').textContent = running ? 'RUNNING' : 'PAUSED'
  const label = running ? 'Pause simulation' : 'Resume simulation'
  if (get('pause').getAttribute('aria-label') !== label) { get('pause').innerHTML = icon(running ? Pause : Play); get('pause').setAttribute('aria-label', label); get('pause').title = label }
  app.dataset.kernel = kernel; app.dataset.precision = precision
}

const values = [-3, -1.2, -0.1, 0, 0.3, 1.4, 3]
const quantized = quantizationExample(values)
get('quantization').innerHTML = `<dl class="facts"><div><dt>Scale</dt><dd>${quantized.scale.toFixed(6)}</dd></div><div><dt>Max reconstruction error</dt><dd>${quantized.maxError.toFixed(6)}</dd></div></dl><table><thead><tr><th>Input</th><th>INT8</th><th>Rebuilt</th></tr></thead><tbody>${values.map((value, index) => `<tr><td>${value.toFixed(2)}</td><td>${quantized.quantized[index]}</td><td>${quantized.reconstructed[index].toFixed(4)}</td></tr>`).join('')}</tbody></table>`

function stopTour() { tourIndex = null; get('tour').hidden = true; get('tour-start').setAttribute('aria-pressed', 'false') }
function showTour(index: number) {
  if (index >= TOUR.length) { stopTour(); switchView('gpu'); city?.home(); return }
  const current = Math.max(0, index); tourIndex = current; tourTime = 0
  const district = DISTRICTS.find((entry) => entry.id === TOUR[current])!
  inspect(district.id, true)
  get('tour').hidden = false; get('tour-start').setAttribute('aria-pressed', 'true')
  get('tour-count').textContent = `GUIDED TOUR / ${current + 1} OF ${TOUR.length}`
  get('tour-title').textContent = district.name; get('tour-body').textContent = district.description
  get<HTMLButtonElement>('tour-prev').disabled = current === 0
  get('tour-progress').innerHTML = TOUR.map((_, index) => `<i class="${index <= current ? 'visited' : ''}"></i>`).join('')
}
function notify(message: string) {
  clearTimeout(noticeTimer); get('notice').textContent = message; get('notice').classList.add('visible')
  noticeTimer = setTimeout(() => get('notice').classList.remove('visible'), 2500)
}

for (const button of document.querySelectorAll<HTMLButtonElement>('[data-view]')) button.addEventListener('click', () => { stopTour(); switchView(button.dataset.view as View) })
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-kernel]')) button.addEventListener('click', () => {
  kernel = button.dataset.kernel as Kernel
  if (kernel !== 'matrix' && precision === 'tf32') precision = 'fp32'
  elapsed = 0; updateModel()
})
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-precision]')) button.addEventListener('click', () => { precision = button.dataset.precision as Precision; elapsed = 0; updateModel() })
for (const input of Object.values(inputs)) input.addEventListener('input', () => { elapsed = 0; updateModel() })
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-tab]')) {
  button.addEventListener('click', () => setTab(button.dataset.tab!))
  button.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault(); const names = ['kernel', 'inspect', 'precision', 'toolkit']
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? 3 : (names.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : 3)) % 4
    setTab(names[index]); get(`tab-${names[index]}`).focus()
  })
}
get('pause').addEventListener('click', () => { running = !running; updateModel() })
get('step').addEventListener('click', () => { running = false; elapsed += 1 / 3; updateModel() })
get('reset').addEventListener('click', () => {
  elapsed = 0; kernel = 'matrix'; precision = 'fp16'; inputs.threads.value = '256'; inputs.registers.value = '32'; inputs.shared.value = '12'
  stopTour(); switchView('gpu'); updateModel()
})
get('home').addEventListener('click', () => { stopTour(); city?.home() })
get('zoom-in').addEventListener('click', () => { stopTour(); city?.zoom(1) })
get('zoom-out').addEventListener('click', () => { stopTour(); city?.zoom(-1) })
get('labels').addEventListener('change', () => city?.setLabels(get<HTMLInputElement>('labels').checked))
get('theme').addEventListener('click', () => {
  day = !day; document.documentElement.dataset.theme = day ? 'day' : 'night'; city?.setDay(day)
  get('theme').innerHTML = icon(day ? Moon : Sun); get('theme').setAttribute('aria-label', day ? 'Switch to night view' : 'Switch to day view'); get('theme').title = get('theme').getAttribute('aria-label')!
})
get('tour-start').addEventListener('click', () => tourIndex === null ? showTour(0) : stopTour())
get('tour-prev').addEventListener('click', () => showTour((tourIndex ?? 0) - 1))
get('tour-next').addEventListener('click', () => showTour((tourIndex ?? 0) + 1))
get('tour-close').addEventListener('click', stopTour)
get('copy-commands').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(get('native-commands').textContent!); notify('Verification commands copied') }
  catch { notify('Clipboard access unavailable') }
})
get('export').addEventListener('click', () => {
  const snapshot = { schema: 'cuda-simcity/v1', kind: 'illustrative-not-measurement', reference: ADA_SM, kernel, precision, resources: { threadsPerBlock: Number(inputs.threads.value), registersPerThread: Number(inputs.registers.value), sharedMemoryKiB: Number(inputs.shared.value) }, capacityBound: residency, modelActivity: sample, elapsed }
  const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'cuda-simulation.json'; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000); notify('Simulation snapshot exported')
})
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { stopTour(); return }
  if (event.target instanceof HTMLElement && (event.target.matches('input, select, textarea, button, a') || event.target.isContentEditable) || event.ctrlKey || event.metaKey || event.altKey) return
  const key = event.key.toLowerCase()
  if (key === ' ' || key === 'p') { event.preventDefault(); get('pause').click() }
  if (key === 't') get('tour-start').click()
  if (key === 'h') get('home').click()
  if (key === 'r') get('reset').click()
  if (key === 'n') get('theme').click()
  if (key === '1' || key === '2') { stopTour(); switchView(key === '1' ? 'gpu' : 'sm') }
})
reduced.addEventListener('change', () => { if (reduced.matches) { running = false; stopTour(); updateModel() } })
switchView('gpu'); updateModel()
const timer = new Timer(); timer.connect(document)
let animationId = 0
let lastUi = -1
function animate(timestamp: number) {
  animationId = requestAnimationFrame(animate); timer.update(timestamp)
  const delta = Math.max(0, Math.min(timer.getDelta(), 0.1))
  if (running) elapsed += delta
  if (Math.floor(elapsed * 10) !== lastUi) { updateModel(); lastUi = Math.floor(elapsed * 10) }
  city?.render(elapsed, sample, kernel, precision, delta)
  if (tourIndex !== null && !reduced.matches) { tourTime += delta; if (tourTime > 7) showTour(tourIndex + 1) }
}
animationId = requestAnimationFrame(animate)
if (import.meta.hot) import.meta.hot.dispose(() => { cancelAnimationFrame(animationId); timer.dispose(); city?.dispose() })