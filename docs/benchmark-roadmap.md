# CUDA SimCity: Benchmark and Iteration Draft

Status: draft, 2026-09-19. The browser/deployment improvements listed below are
implemented; the remaining feature and model work is proposed.

## Reference Standard

Use HexagonNPUSimCity for explicit illustrative-versus-measured boundaries,
reviewed configuration, and production-browser verification. Use PGSimCity for
pure simulation ownership, deterministic mechanism tests, and causal scenarios.
These are quality references, not templates to copy wholesale.

## Observed Baseline

- [Simulation](../src/sim/model.ts): Ada compute-capability 8.9 residency limits,
  whole-block admission, explicit SIMT/MMA routing, TF32 storage semantics, and
  an INT8 quantization example. Kernel utilization remains a teaching profile.
- [Model tests](../src/sim/model.test.ts): resource limits, partial warps,
  invalid launches, idle behavior, precision routing, and quantization errors.
- [Architecture registry](../src/sim/architecture.ts): shared district facts,
  source URLs, tour order, and explicit per-district limits already exist.
- [Application](../src/main.ts): GPU/SM views, inspector, tour, precision lab,
  reduced-motion handling, step/reset, keyboard tabs, snapshot export, and a
  non-WebGL fallback. Allocation-granularity omissions are already stated in UI.
- [Project commands](../package.json): typecheck, unit tests, build, and a
  Playwright command. At the start of this pass that browser command discovered
  zero tests; the new suite makes it executable coverage.
- [README](../README.md): discloses a simplified model, but does not provide a
  claim-by-claim source and simplification audit.

## Landed In This Pass

- [Browser suite](../tests/app.spec.mjs) and [configuration](../playwright.config.ts):
  production build under a non-root path, desktop/mobile GPU and SM canvas
  pixels/framing/containment, rendered changes after stepping, resource controls,
  precision/idle behavior, pause/step/reset, tour, keyboard tabs, and labeled
  export. Four browser tests, ten unit tests, typecheck, and build pass locally.
- [Vite configuration](../vite.config.ts): relative production asset URLs. The
  original absolute `/assets/` references failed the subpath boot check; the
  same browser check passed after the fix.
- [Pages workflow](../.github/workflows/pages.yml): installs Chromium and runs
  the browser suite before uploading a deployment artifact. This workflow
  change has been checked locally, not executed on GitHub in this pass.

## Next Iteration: Verifiable Occupancy

Hypothesis: the model can teach residency without implying GPU performance if
architectural limits, omitted allocation rules, and illustrative utilization
coefficients are separately documented and tested.

1. Add a verification ledger linking every `ADA_SM` limit to the appropriate
   version of NVIDIA's CUDA Programming Guide or Ada Tuning Guide. Record units,
   scope, source section, and review date.
2. Carry the existing UI disclosures about register-allocation granularity and
  shared-memory assumptions into that ledger. Keep calling the output a
  capacity bound, not a hardware occupancy oracle; adding allocation rules
  later requires independently verified architecture-specific fixtures.
3. Keep theoretical resident warps separate from achieved occupancy, eligible
   warps, issue rate, and measured utilization. Do not derive speedup from
   occupancy alone.
4. Add boundary fixtures around register-allocation and block-admission changes,
   then compare supported cases with an independently obtained occupancy result.
   A retained external result must identify GPU, CUDA version, kernel attributes,
   launch parameters, and capture method.

Cheap discriminating check: independently compute one allocation-boundary case
and assert the admitted block count; existing arithmetic-only tests cannot prove
that an omitted hardware allocation rule is harmless.

Acceptance: each displayed hardware fact has a source; every model coefficient
is labeled illustrative; invalid launches cannot animate productive work; a
production-bundle browser test exercises the same controls as the model tests.

## Candidate Features

- **P1: memory transactions.** Derive addresses and touched memory segments for
  contiguous, strided, and permuted accesses; expose a causal transaction count.
  Start with a declared architecture and operation, not a universal bandwidth
  estimate. Test exact address-to-segment mappings.
- **P1: block scheduling timeline.** Explain resident versus queued blocks and
  resource release. A deterministic launch fixture must show why increasing
  registers or shared memory reduces admitted blocks.
- **P2: roofline experiment.** Pair explicit operation and byte counts with
  user-supplied, provenance-labeled ceilings. Keep the bound distinct from a
  performance prediction; no default model result may be called measured.
- **P2: additional GPU profiles.** Consider Ampere and Hopper only after each
  has an independent capability table, unsupported-operation behavior, primary
  sources, and profile-specific boundary fixtures. Do not make a new city solely
  to change SM counts or colors.

## Delivery Gates

Run `npm test`, `npm run typecheck`, and `npm run build` for implementation work.
Run `npm run test:browser` for rendered changes, including desktop and mobile,
pause/reset, input boundaries, and a production subpath. Record actual results
separately from this draft. Hardware comparisons are optional follow-up evidence,
not a prerequisite for running the educational model offline.