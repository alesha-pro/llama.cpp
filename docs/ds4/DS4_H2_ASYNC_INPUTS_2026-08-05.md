# H2: async input uploads (DSV4_ASYNC_INPUTS) — implemented, measured NO-OP

Date: 2026-08-05. Follow-up to DS4_HPROF_DECODE_SYNCS_2026-08-05.md.

## Hypothesis

`dsv4_graph_inputs::set_input` issues ~60-80 blocking `ggml_backend_tensor_set`
per token (per-layer scalar indices + attention masks); the nsys capture
suggested ~66 host-blocking `cudaStreamSynchronize` per token at ~257 us each.

## Change

- `ggml_backend_cuda_tensor_set_input_async` (ggml-cuda): async H2D on the home
  device's compute stream via a device->backend-context registry; falls back to
  the blocking set for non-CUDA buffers.
- `dsv4_graph_inputs::set_input`: all uploads gathered into a ringed staging
  arena (8 generations; the host may run up to the sched pipeline depth ahead
  of the device), enqueued without host sync; decode masks (n1 == 1) get
  content versioning — RAW_WINDOW content is constant, COMPRESS_* depends only
  on `n_visible = (last_pos+1)/ratio` — unchanged masks skip fill+upload.
- Flag: `DSV4_ASYNC_INPUTS=1` (opt-in).

## Result

**No speed change anywhere.** Short ctx: 42.7 vs 42.8 t/s. Deep ctx (~130K):
39.37 vs 39.35 t/s (A/B same night, same server config). A shim with per-call
timing then showed the truth the nsys capture had distorted:

- `ggml_backend_cuda_buffer_set_tensor` syncs: **2.4 us each** (51,696 calls,
  124 ms total) — the input uploads were cheap all along; nsys instrumentation
  had inflated them to 257 us.
- The real cost sits in `ggml_backend_cuda_synchronize` called from
  `ggml_backend_sched_synchronize`: **785 us average**, ~92% of decode wall.

Also established by direct microbenchmark on this rig/driver: an idle
`cudaStreamSynchronize` costs **0.27 us** — the 275+ us syncs are waiting for
real queued work, not paying syscall/driver overhead.

## Verdict

Keep the code behind the flag (off by default): it is correct and removes
API-call count, but it is not a decode lever. The mask content-versioning part
is harmless and stays active only with the flag. Do not spend more time here.

## Lesson

nsys inflates per-CUDA-call costs on this box (custom 610.43.02 P2P driver);
always cross-check with an LD_PRELOAD timing shim before believing per-call
attribution. GPU telemetry (nvidia-smi dmon) showed the real picture:
SM util 20-28%, ~200 W per GPU during decode — the GPUs ARE working; decode
is a latency-chained GPU workload, not an idle-GPU host-bound one.
