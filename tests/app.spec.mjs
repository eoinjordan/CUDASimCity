import { test, expect } from '@playwright/test'
import { PNG } from 'pngjs'
import { readFile } from 'node:fs/promises'

async function open(page) {
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto('./')
  await expect(page.locator('#scene > canvas')).toBeVisible()
  await expect(page.locator('#run-state')).toHaveText('PAUSED')
  return errors
}

async function canvasPixels(page) {
  return PNG.sync.read(await page.locator('#scene > canvas').screenshot({
    style: '#scene > :not(canvas), .scene-heading, .camera-tools, .legend, #tour { visibility: hidden !important; }',
  }))
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`production layout renders at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    const errors = await open(page)
    for (const view of ['gpu', 'sm']) {
      await page.locator(`button[data-view="${view}"]`).click()
      await expect(page.locator(`button[data-view="${view}"]`)).toHaveAttribute('aria-pressed', 'true')
      await expect(page.locator('#scene')).toHaveAttribute('data-view', view)
      await expect(page.locator('#scene')).toHaveAttribute('data-framed', '6')
      const pixels = await canvasPixels(page)
      const colors = new Set()
      let colored = 0
      for (let index = 0; index < pixels.data.length; index += 64) {
        const [red, green, blue] = pixels.data.subarray(index, index + 3)
        colors.add(`${red >> 4},${green >> 4},${blue >> 4}`)
        if (Math.max(red, green, blue) - Math.min(red, green, blue) > 30) colored += 1
      }
      expect(colors.size).toBeGreaterThan(40)
      expect(colored).toBeGreaterThan(80)
      expect(pixels.width).toBeGreaterThan(300)
      expect(pixels.height).toBeGreaterThan(200)
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      await page.screenshot({ path: test.info().outputPath(view === 'gpu' ? 'overview.png' : 'sm-overview.png'), fullPage: true })
    }
    expect(errors).toEqual([])
  })
}

test('model controls preserve capacity, precision, pause, and reset contracts', async ({ page }) => {
  const errors = await open(page)
  const before = await canvasPixels(page)
  await page.locator('#step').click()
  const after = await canvasPixels(page)
  let differences = 0
  for (let index = 0; index < before.data.length; index += 4) {
    if (Math.abs(before.data[index] - after.data[index]) > 10
      || Math.abs(before.data[index + 1] - after.data[index + 1]) > 10) differences += 1
  }
  expect(differences).toBeGreaterThan(100)
  await page.locator('#reset').click()
  await page.locator('#threads').focus()
  await page.keyboard.press('End')
  await page.locator('#registers').focus()
  await page.keyboard.press('End')
  await expect(page.locator('#blocks')).toHaveText('0')
  await expect(page.locator('#engine')).toHaveText('NO ACTIVE WORK')
  await page.locator('#reset').click()
  await page.locator('[data-precision="tf32"]').click()
  await expect(page.locator('#tile-storage')).toHaveText('32KiB')
  await page.locator('[data-kernel="reduction"]').click()
  await expect(page.locator('#app')).toHaveAttribute('data-precision', 'fp32')
  await expect(page.locator('[data-precision="tf32"]')).toBeDisabled()
  await page.locator('[data-kernel="idle"]').click()
  await expect(page.locator('#engine')).toHaveText('NO ACTIVE WORK')
  await expect(page.locator('#resident-count')).toHaveText('0 / 48')
  await page.locator('#step').click()
  await expect(page.locator('#clock')).toHaveText('0.3s')
  await page.locator('#theme').click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'day')
  await expect(page.locator('#clock')).toHaveText('0.3s')
  await page.locator('#reset').click()
  await expect(page.locator('#clock')).toHaveText('0.0s')
  await expect(page.locator('#app')).toHaveAttribute('data-kernel', 'matrix')
  expect(errors).toEqual([])
})

test('views, tour, keyboard tabs, and export preserve architectural context', async ({ page }) => {
  const errors = await open(page)
  await page.locator('button[data-view="sm"]').click()
  await expect(page.locator('#scene-title')).toHaveText('Inside a streaming multiprocessor')
  await page.locator('#tour-start').click()
  await expect(page.locator('#tour-title')).toHaveText('Host CPU')
  await page.locator('#tour-next').click()
  await expect(page.locator('#tour-title')).toHaveText('Grid dispatch')
  await page.keyboard.press('Escape')
  await expect(page.locator('#tour')).toBeHidden()
  await page.getByRole('tab', { name: 'Kernel', exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Inspect', exact: true })).toBeFocused()
  const downloaded = page.waitForEvent('download')
  await page.locator('#export').click()
  const artifact = await downloaded
  const snapshot = JSON.parse(await readFile(await artifact.path(), 'utf8'))
  expect(snapshot.schema).toBe('cuda-simcity/v1')
  expect(snapshot.kind).toBe('illustrative-not-measurement')
  expect(snapshot.reference.warpSize).toBe(32)
  expect(snapshot.resources.threadsPerBlock).toBe(256)
  expect(errors).toEqual([])
})