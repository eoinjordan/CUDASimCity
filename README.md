# CUDA SimCity

An interactive 3D explainer for CUDA-style parallel execution. Adjust block size, register pressure, and shared-memory use to see how an educational occupancy model changes active warps across a city of streaming multiprocessors.

![CUDA SimCity live preview](docs/media/demo.gif)

The model is intentionally simplified and does not claim specifications or measured performance for any NVIDIA GPU.

## Development

```bash
npm install
npm run dev
npm run build
```