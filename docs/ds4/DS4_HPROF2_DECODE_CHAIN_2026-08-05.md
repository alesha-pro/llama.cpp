# H-prof2: what actually bounds decode — the 21 ms GPU chain

Date: 2026-08-05. Supersedes the sync-centric reading of
DS4_HPROF_DECODE_SYNCS_2026-08-05.md (nsys per-call numbers were inflated).

## Method

- LD_PRELOAD timing shim on cudaStreamSynchronize/cudaGraphLaunch/cudaMemcpyAsync
  (no profiler distortion): /tmp/syncshim2.c, /tmp/syncshim3.c (backtrace
  attribution), SYNCLOG caller tagging in llama-context/ggml-backend
  (`__builtin_return_address` + addr2line).
- `perf record` on the server during decode (sudo, -F 1999).
- `nvidia-smi dmon` GPU telemetry during a decode burst.
- CUDA microbenchmark: idle streamSync on this driver = **0.27 us**
  (/tmp/syncbench.cu) — sync calls themselves are cheap.

## Findings

Per-token budget: 23.4 ms (42.8 t/s, short ctx, PL 270, -lgc 1995).

- Per token the server issues ~7 `llama_context::synchronize()` calls
  (process_ubatch + llama_get_logits_ith + llama_get_sampled_{token,probs,
  logits,candidates}_ith + common_sampler_sample). Instrumented durations:
  **one** of them waits **21.0 ms**; all others are 5-9 us. The redundant
  syncs are FREE — there is nothing to save here.
- The 21 ms wait is the genuine GPU critical path of one decode step:
  43 layers serialized across 4 GPUs (layer split), ~5.9 ms of kernel time
  per GPU per token, SM util 20-28%, mem util 11-16%, ~200 W per GPU
  (measured by dmon during a 256-token burst).
- Per layer: ~490 us measured vs ~60-90 us of weight-read floor
  (6 routed experts ~40 MB + shared + attention projections at 936 GB/s).
  The gap is serialized small-kernel latency along the layer chain.
- CUDA graphs DO replay in steady decode (RUDBG: 45/52 reuse=1; the rest are
  legit request transitions). The earlier "graphs never replay" reading was
  a CUPTI artifact: graph-contained kernels are invisible in the kernel
  table of this nsys version, which also faked the "GPU busy 0.5%" number.

## Consequences

- Decode is a latency-bound GPU chain. Two rational levers:
  1. **Amortize the chain**: speculative decode (H1/DSpark) — a 2-token
     verify costs ~the same 21 ms chain, so acceptance directly multiplies
     throughput.
  2. **Shorten the chain**: fewer/faster kernels per layer (fusion, better
     GEMV kernels for IQ2_XXS/IQ3_XXS on Ampere — see KernelWiki), or
     cross-layer overlap. Bigger surgery, no quick win identified tonight.
- NOT levers (measured): input upload syncs (H2), redundant ctx syncs,
  clock locks beyond droop removal (H0), power limit (decode is not
  power-bound).
