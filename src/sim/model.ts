export type Kernel = 'matrix' | 'reduction' | 'stencil' | 'idle'
export type Precision = 'fp32' | 'tf32' | 'fp16' | 'int8'

export const PRECISIONS = {
  fp32: { name: 'FP32', bytes: 4, accumulator: 'FP32', description: 'SIMT FP32 path in this example.' },
  tf32: { name: 'TF32', bytes: 4, accumulator: 'FP32', description: 'Tensor math mode with FP32 storage; not a 19-bit storage format.' },
  fp16: { name: 'FP16', bytes: 2, accumulator: 'FP32', description: 'Explicit matrix MMA path with FP16 inputs and FP32 accumulation.' },
  int8: { name: 'INT8', bytes: 1, accumulator: 'INT32', description: 'Explicit integer matrix path; scales and calibration remain necessary.' },
} as const

export const ADA_SM = {
  name: 'Ada / compute capability 8.9',
  warpSize: 32,
  maxWarps: 48,
  maxBlocks: 24,
  maxThreadsPerBlock: 1024,
  registers: 65536,
  maxRegistersPerThread: 255,
  sharedMemoryKiB: 100,
  maxSharedMemoryPerBlockKiB: 99,
  reservedSharedMemoryPerBlockKiB: 1,
} as const

const kernels = {
  matrix: { arithmetic: 0.92, coalescing: 0.88, divergence: 0.03 },
  reduction: { arithmetic: 0.62, coalescing: 0.76, divergence: 0.18 },
  stencil: { arithmetic: 0.45, coalescing: 0.58, divergence: 0.08 },
  idle: { arithmetic: 0, coalescing: 0, divergence: 0 },
} as const

export function executionPath(kernel: Kernel, precision: Precision) {
  if (kernel === 'idle') return { engine: 'none' as const, instruction: 'No kernel', rtActive: false }
  if (kernel === 'matrix' && precision !== 'fp32') return { engine: 'tensor' as const, instruction: `${PRECISIONS[precision].name} MMA / ${PRECISIONS[precision].accumulator} accumulator`, rtActive: false }
  return { engine: 'cuda' as const, instruction: kernel === 'matrix' ? 'FP32 SIMT multiply-accumulate' : kernel === 'reduction' ? 'SIMT arithmetic + warp reduction' : 'SIMT loads + stencil arithmetic', rtActive: false }
}

export function storageBytes(elements: number, precision: Precision): number {
  if (!Number.isSafeInteger(elements) || elements < 0 || !PRECISIONS[precision]) throw new RangeError('Invalid tensor storage request')
  return elements * PRECISIONS[precision].bytes
}

export function quantizeInt8(value: number, scale: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(scale) || scale <= 0) throw new RangeError('Quantization requires a finite value and positive scale')
  const scaled = value / scale
  const lower = Math.floor(scaled)
  const fraction = scaled - lower
  const rounded = fraction === 0.5 ? lower + Math.abs(lower % 2) : Math.round(scaled)
  return Math.max(-128, Math.min(127, rounded))
}

export function quantizationExample(values: readonly number[]) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) throw new RangeError('Expected a nonempty finite tensor')
  const scale = Math.max(...values.map(Math.abs)) / 127 || 1
  const quantized = values.map((value) => quantizeInt8(value, scale))
  const reconstructed = quantized.map((value) => value * scale)
  const maxError = Math.max(...values.map((value, index) => Math.abs(value - reconstructed[index])))
  return { scale, zeroPoint: 0, quantized, reconstructed, maxError }
}

export function calculateResidency(threadsPerBlock: number, registersPerThread: number, sharedMemoryKiB: number) {
  const valid = Number.isInteger(threadsPerBlock) && threadsPerBlock > 0 && threadsPerBlock <= ADA_SM.maxThreadsPerBlock
    && Number.isInteger(registersPerThread) && registersPerThread > 0 && registersPerThread <= ADA_SM.maxRegistersPerThread
    && Number.isFinite(sharedMemoryKiB) && sharedMemoryKiB >= 0 && sharedMemoryKiB <= ADA_SM.maxSharedMemoryPerBlockKiB
  const warpsPerBlock = valid ? Math.ceil(threadsPerBlock / ADA_SM.warpSize) : 0
  const limits = {
    'Warp slots': valid ? Math.floor(ADA_SM.maxWarps / warpsPerBlock) : 0,
    'Block slots': valid ? ADA_SM.maxBlocks : 0,
    Registers: valid ? Math.floor(ADA_SM.registers / (warpsPerBlock * ADA_SM.warpSize * registersPerThread)) : 0,
    'Shared memory': valid ? Math.floor(ADA_SM.sharedMemoryKiB / (sharedMemoryKiB + ADA_SM.reservedSharedMemoryPerBlockKiB)) : 0,
  }
  const blocksPerSm = Math.min(...Object.values(limits))
  const residentWarps = blocksPerSm * warpsPerBlock
  return {
    valid,
    warpsPerBlock,
    blocksPerSm,
    residentWarps,
    occupancy: residentWarps / ADA_SM.maxWarps,
    limits,
    limitingResources: Object.entries(limits).filter(([, blocks]) => blocks === blocksPerSm).map(([name]) => name),
  }
}

export function calculateOccupancy(threadsPerBlock: number, registersPerThread: number, sharedMemoryKiB: number): number {
  return calculateResidency(threadsPerBlock, registersPerThread, sharedMemoryKiB).occupancy
}

export function sampleKernel(kernel: Kernel, occupancy: number, tick: number) {
  const profile = kernels[kernel]
  const active = kernel === 'idle' ? 0 : Number.isFinite(occupancy) ? Math.max(0, Math.min(1, occupancy)) : 0
  const sampleTick = Number.isFinite(tick) ? Math.max(0, Math.floor(tick)) : 0
  const wave = 0.94 + ((sampleTick * 17) % 11) / 100
  return {
    activeWarps: Math.round(ADA_SM.maxWarps * active),
    computeUse: Math.min(1, profile.arithmetic * active * wave),
    memoryUse: Math.min(1, (1 - profile.arithmetic * 0.45) * profile.coalescing * active * wave),
    divergence: active > 0 ? profile.divergence : 0,
  }
}