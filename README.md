# CUDA SimCity

An interactive 3D explainer for CUDA-style parallel execution. Adjust block size, register pressure, and shared-memory use to see how an educational occupancy model changes active warps across a city of streaming multiprocessors.

[Open CUDA SimCity](https://eoinjordan.github.io/CUDASimCity/).

![CUDA SimCity live preview](docs/media/demo.gif)

This is an independent educational preview, not GPU telemetry or a hardware benchmark. The city does not execute CUDA or connect to an NVIDIA GPU.

## Current Features

- GPU overview and an SM detail view, with source-linked district inspection and a guided architecture tour.
- Whole-block residency from threads per block, registers per thread, and shared memory per block using an Ada compute-capability 8.9 reference.
- Matrix, reduction, stencil, and idle teaching workloads; explicit SIMT versus matrix-instruction routing instead of assuming that low precision always uses Tensor Cores.
- FP32, TF32, FP16, and INT8 explanations, tile-storage accounting, and an INT8 quantization example. TF32 retains FP32 storage.
- Pause, single-step, reset, theme controls, keyboard tabs, and a JSON snapshot labeled `illustrative-not-measurement`.

The [model](src/sim/model.ts), [architecture registry](src/sim/architecture.ts), and [tests](src/sim/model.test.ts) are the implementation references. The registry links architectural claims to their sources and states district-specific limits.

## Model Limits

Published Ada limits inform a simplified per-SM capacity bound. Register-allocation granularity, partition constraints, and architecture-specific allocation rules are omitted. Resident warps are not measured occupancy, issue rate, utilization, or a performance prediction. Kernel activity and packet movement remain illustrative.

The scene is not a die floorplan or a CUDA kernel debugger. It does not compile Triton or CUDA Tile programs, model address-derived memory transactions, or provide Ampere/Hopper profiles. Those are proposed lessons in the [benchmark roadmap](docs/benchmark-roadmap.md), not current capabilities.

## Recorded Preview

The refreshed GIF visits the GPU and SM views, raises register pressure to reduce block capacity from six to two, rejects a 1024-thread launch at 128 registers per thread, and resets before comparing TF32 and INT8 storage. The recorder asserts those outcomes, including zero active work for the impossible launch.

The 24 frames play at 3 fps with edited model progression, not GPU timing. [Recording metadata](docs/media/recording.json) retains the capture date, source/build/GIF hashes, observed controls and metrics, and desktop/mobile checks.

## Verification

The retained suite has ten model tests and four production-browser tests. Browser coverage includes desktop/mobile GPU and SM views, canvas pixels and framing, resource and precision controls, pause/step/reset, tour navigation, keyboard tabs, and snapshot export under a non-root URL. These checks do not validate native GPU execution or hardware occupancy.

## Development

```bash
npm ci
npm run dev
npm test
npm run typecheck
npx playwright install chromium
npm run test:browser
npm run build
```