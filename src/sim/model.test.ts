import assert from 'node:assert/strict'
import test from 'node:test'
import { ADA_SM, calculateOccupancy, calculateResidency, sampleKernel, executionPath, storageBytes, quantizeInt8, quantizationExample } from './model.ts'

test('resource pressure lowers occupancy', () => {
  const light = calculateOccupancy(256, 24, 8)
  const heavy = calculateOccupancy(256, 128, 96)
  assert.equal(light, 1)
  assert.equal(heavy, 8 / 48)
})

test('kernel samples remain normalized', () => {
  const sample = sampleKernel('matrix', 0.75, 4)
  assert.equal(sample.activeWarps, 36)
  assert.ok(sample.computeUse >= 0 && sample.computeUse <= 1)
  assert.ok(sample.memoryUse >= 0 && sample.memoryUse <= 1)
})

test('register pressure limits the number of whole resident blocks', () => {
  const result = calculateResidency(256, 128, 0)
  assert.equal(result.blocksPerSm, 2)
  assert.equal(result.residentWarps, 16)
  assert.equal(result.occupancy, 1 / 3)
  assert.deepEqual(result.limitingResources, ['Registers'])
})

test('shared memory includes the Ada per-block reservation', () => {
  assert.equal(calculateResidency(256, 32, 49).blocksPerSm, 2)
  assert.equal(calculateResidency(256, 32, 50).blocksPerSm, 1)
  assert.equal(calculateResidency(256, 32, 99).blocksPerSm, 1)
})

test('partial warps and block slots constrain residency', () => {
  assert.equal(calculateResidency(33, 32, 0).warpsPerBlock, 2)
  const result = calculateResidency(32, 16, 0)
  assert.equal(result.blocksPerSm, ADA_SM.maxBlocks)
  assert.equal(result.occupancy, 0.5)
})

test('invalid or unschedulable launches have zero occupancy', () => {
  for (const args of [[0, 32, 0], [1025, 32, 0], [256, 0, 0], [256, 256, 0], [256, 32, 100], [256, 32, -1], [NaN, 32, 0], [256, Infinity, 0], [256, 32, NaN], [64.5, 32, 0], [1024, 128, 0]]) {
    assert.equal(calculateOccupancy(args[0], args[1], args[2]), 0)
  }
})

test('tensor instructions are explicit and RT cores are not generic CUDA arithmetic', () => {
  assert.equal(executionPath('matrix', 'fp32').engine, 'cuda')
  for (const precision of ['tf32', 'fp16', 'int8'] as const) assert.equal(executionPath('matrix', precision).engine, 'tensor')
  for (const kernel of ['matrix', 'reduction', 'stencil', 'idle'] as const) {
    assert.equal(executionPath(kernel, 'fp16').rtActive, false)
  }
  assert.equal(executionPath('reduction', 'fp16').engine, 'cuda')
  assert.equal(executionPath('idle', 'fp16').engine, 'none')
})

test('TF32 math does not shrink FP32 storage', () => {
  assert.equal(storageBytes(1024, 'tf32'), 4096)
  assert.equal(storageBytes(1024, 'fp32'), 4096)
  assert.equal(storageBytes(1024, 'fp16'), 2048)
  assert.equal(storageBytes(1024, 'int8'), 1024)
  assert.throws(() => storageBytes(-1, 'int8'), RangeError)
})

test('INT8 reference uses nearest-even rounding, clipping, and positive scales', () => {
  assert.equal(quantizeInt8(2.5, 1), 2)
  assert.equal(quantizeInt8(3.5, 1), 4)
  assert.equal(quantizeInt8(-2.5, 1), -2)
  assert.equal(quantizeInt8(-3.5, 1), -4)
  assert.equal(quantizeInt8(1000, 1), 127)
  assert.equal(quantizeInt8(-1000, 1), -128)
  assert.throws(() => quantizeInt8(1, 0), RangeError)
  const result = quantizationExample([-3, -1.2, -0.1, 0, 0.3, 1.4, 3])
  assert.ok(result.maxError <= result.scale / 2 + Number.EPSILON)
  assert.deepEqual(quantizationExample([0, 0]).quantized, [0, 0])
})

test('idle and impossible launches have no simulated work or memory flow', () => {
  for (const sample of [sampleKernel('idle', 1, 1), sampleKernel('matrix', 0, 1), sampleKernel('matrix', NaN, 1)]) {
    assert.equal(sample.activeWarps, 0)
    assert.equal(sample.computeUse, 0)
    assert.equal(sample.memoryUse, 0)
  }
  assert.ok(sampleKernel('matrix', 10, Infinity).computeUse <= 1)
})