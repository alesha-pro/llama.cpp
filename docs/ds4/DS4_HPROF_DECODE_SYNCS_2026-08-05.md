# H-prof: decode profile — 99 host-blocking stream syncs per token

Date: 2026-08-05. Rig: 4x RTX 3090, UD-IQ2_M, CTX=131072, prod flags, PL 270, -lgc 1995.

## Method

nsys profile of llama-server during two 256-token decode bursts
(/tmp/ds4-decode-prof.nsys-rep), LD_PRELOAD shim on cudaStreamSynchronize for
callsite attribution (/tmp/syncshim.c), addr2line on libggml-cuda.so.

## Findings

Per-token budget at baseline: 23.3 ms (42.8 t/s short ctx).

- **Each GPU is busy 0.5% of wall.** Total kernel time 0.127 s of a 6.5 s
  decode window (all 4 GPUs summed). Decode is ~98% host-side overhead.
- **25,431 cudaStreamSynchronize in 6.5 s = 6,540 ms ≈ 100% of wall.**
  ~99 syncs per token, avg 257 us each.
- Callsite attribution (shim + addr2line):
  - ~2/3: `ggml_backend_cuda_buffer_set_tensor` — the blocking input upload
    (cudaMemcpyAsync + cudaStreamSynchronize). Sources: `dsv4_graph_inputs::
    set_input` in src/models/deepseek4.cpp writes ~60 deduped per-layer scalar
    index inputs + several attention masks PER TOKEN, each with its own
    full-stream host sync.
  - ~1/3: `ggml_backend_cuda_synchronize` — sched/context-level syncs
    (ggml_backend_sched_synchronize loops all 5 backends per graph compute,
    ~6 computes/token).
- 21,239 cudaMemcpyAsync (83/token), 1,264 cudaGraphLaunch (~5/token),
  42,640 loose cudaLaunchKernel (166/token).
- Top kernels are all tiny: mul_mat_vec_q 18-44 us, sinkhorn 14 us,
  fp8_kv_quantize 17 us — per-GPU per-token work is ~0.12 ms.

## Consequences

- Decode is latency-bound, not bandwidth- or compute-bound. The theoretical
  headroom is large; the two sync classes are the first targets.
- H2 attacks the input-upload class (~66 syncs/token): async uploads on the
  home device's compute stream + decode-mask content versioning.
- H3 (later): the sched/ctx sync class (~33/token).
- The earlier REPEAT-I32 fix (DS4HANDOFF 0d) was the same class of problem;
  this is the remaining tail of it.

## Repro

```
nsys profile -t cuda -o /tmp/ds4-decode-prof --force-overwrite=true \
  bash scripts/ds4-prod-serve.sh   # WARM=0
# then 2x 256-token decode bursts, kill server
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_sum ds4-decode-prof.nsys-rep
```
