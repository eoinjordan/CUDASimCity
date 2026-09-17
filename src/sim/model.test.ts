import assert from 'node:assert/strict'
import test from 'node:test'
import { calculateOccupancy, sampleKernel } from './model.ts'

test('resource pressure lowers occupancy', () => {
  const light = calculateOccupancy(256, 24, 8)
  const heavy = calculateOccupancy(256, 128, 96)
  assert.ok(light > heavy)
  assert.ok(heavy >= 0.125)
})

test('kernel samples remain normalized', () => {
  const sample = sampleKernel('matrix', 0.75, 4)
  assert.equal(sample.activeWarps, 48)
  assert.ok(sample.computeUse >= 0 && sample.computeUse <= 1)
  assert.ok(sample.memoryUse >= 0 && sample.memoryUse <= 1)
})