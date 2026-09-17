export type Kernel = 'matrix' | 'reduction' | 'stencil'

const kernels = {
  matrix: { arithmetic: 0.92, coalescing: 0.88, divergence: 0.03 },
  reduction: { arithmetic: 0.62, coalescing: 0.76, divergence: 0.18 },
  stencil: { arithmetic: 0.45, coalescing: 0.58, divergence: 0.08 },
} as const

export function calculateOccupancy(threadsPerBlock: number, registersPerThread: number, sharedMemoryKb: number): number {
  const threadLimit = Math.floor(2048 / threadsPerBlock) * threadsPerBlock / 2048
  const registerLimit = Math.min(1, 65536 / (registersPerThread * Math.max(threadsPerBlock, 1)))
  const sharedLimit = sharedMemoryKb === 0 ? 1 : Math.min(1, 96 / sharedMemoryKb / 8)
  return Math.max(0.125, Math.min(1, threadLimit, registerLimit, sharedLimit))
}

export function sampleKernel(kernel: Kernel, occupancy: number, tick: number) {
  const profile = kernels[kernel]
  const wave = 0.94 + ((tick * 17) % 11) / 100
  return {
    activeWarps: Math.round(64 * occupancy),
    computeUse: profile.arithmetic * occupancy * wave,
    memoryUse: (1 - profile.arithmetic * 0.45) * profile.coalescing * wave,
    divergence: profile.divergence,
  }
}