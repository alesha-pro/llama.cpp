# DeepSeek-V4 expert parallelism — 2026-07-10

## Result

Implemented an opt-in hybrid expert-parallel path for the private DeepSeek-V4 fork. Attention, KV, shared experts, and residual work keep the existing layer placement. Only the three routed-expert tensors are split across the four GPUs on the expert axis (64 of 256 experts per GPU).

Enable it with:

```bash
DSV4_EXPERT_PARALLEL=1 GGML_CUDA_P2P=1 \
DSV4_CONSTANT_SHAPE=1 DSV4_MOE_TILE=1 DSV4_MOE_RESIDENT=1 \
DSV4_GLU_FUSE=1 DSV4_MOE_FUSE=1 DSV4_DECODE_FUSED_IDX=1 \
build-v4-cuda/bin/llama-batched-bench ... -sm layer
```

The mode is deliberately off by default. It requires `--split-mode layer` and at least two direct GPU devices.

## What was implemented

- Routed `ffn_up_exps`, `ffn_gate_exps`, and `ffn_down_exps` weights through a dedicated meta device while leaving normal layer placement unchanged.
- Split the 256-expert axis equally across all GPUs and retained global router IDs.
- Added local-ID compaction for CUDA `MUL_MAT_ID`, including graph warmup shapes and duplicate expert IDs.
- Kept remote top-k slots zero locally, summed the six local expert outputs, and performed one hidden-state AllReduce per MoE layer.
- Fixed a pre-existing meta-backend corner case where a delayed AllReduce at the end of a subgraph was skipped. Before the fix, only GPU 0's expert contribution reached the next layer.
- Added direct CUDA-to-meta broadcast and mirrored-meta-to-CUDA copies. This removed CPU staging at the hybrid graph boundary: Nsight dropped from 41 Device-to-Host copies to the normal 9 small copies.

## Correctness

Real-weight four-layer fixture:

`/mnt/nvme/engines/llama.cpp/build-ds4-cuda128/ds4-one-layer.gguf`

For the same 18-token prompt, layer split and four-GPU expert parallelism produced:

- identical top-1 token (`92132`);
- logits cosine similarity `0.999944857452`;
- relative L2 difference `0.0105034588`;
- max absolute logit difference `0.21447134`.

The remaining difference is expected from the changed FP accumulation order and the meta backend's BF16 NCCL reduction for large tensors. CUDA build, graph warmup, prefill, and decode smoke tests all completed without illegal accesses or NaNs.

## Performance at 4 x RTX 3090, 220 W

Same binary, fixture, `ubatch=512`, FlashAttention, P2P enabled, and all existing DS4 optimizations enabled.

### Standalone pp256 + tg32, two runs

| Mode | pp256 average | tg32 average | Change |
|---|---:|---:|---:|
| Layer split | 2806.84 t/s | 136.13 t/s | baseline |
| Expert parallel | 3246.95 t/s | 131.51 t/s | prefill `+15.68%`, decode `-3.39%` |

End-to-end combined speed for this short mixed test improved only about `+1.27%` because decode gives back part of the prefill win.

### Repeated pp512 (graph/cache warmup behavior)

| Iteration | Layer split | Expert parallel | EP change |
|---:|---:|---:|---:|
| 1 | 4396.17 | 4448.77 | `+1.20%` |
| 2 | 5930.18 | 5661.15 | `-4.54%` |
| 3 | 6838.34 | 7003.63 | `+2.42%` |
| Average | 5721.56 | 5704.51 | `-0.30%` |

On the four-layer fixture, expert parallelism helps early/cold prefill, but steady pp512 throughput is effectively tied with the optimized device-resident layer path. This shallow fixture conclusion is superseded below by the completed full-model measurement.

## Full-model run

The production GGUF was copied to NVMe so the previously blocked full-model A/B could be completed:

`/mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

Source and destination sizes both verified as `86,720,111,488` bytes. The expert-parallel model loaded successfully across all four RTX 3090s in about 32 seconds; there was no OOM or CPU fallback.

Same clean binary, `ubatch=512`, P2P enabled, all production DS4 flags, no MTP:

| run | layer split prefill | expert parallel prefill | change |
|---|---:|---:|---:|
| pp512 only | 584.80 t/s | 701.03 t/s | `+19.88%` |
| pp512 + tg128 | 577.35 t/s | 703.49 t/s | `+21.85%` |
| mean | 581.08 t/s | 702.26 t/s | **`+20.86%`** |

The mixed run also measured steady decode over 128 tokens:

| mode | decode throughput | latency/token | change |
|---|---:|---:|---:|
| layer split | 36.34 t/s | 27.52 ms | baseline |
| expert parallel | 23.82 t/s | 41.98 ms | **`-34.44%`** |

This is the first completed production-weight full-model measurement. It confirms that four-way expert sharding gives a large real prefill gain, but also that one 4-GPU broadcast/AllReduce per MoE layer is too expensive for interactive decode.

### Two-GPU pair feasibility check

A real-weight four-layer fixture was also run on only two GPUs to model a possible `(GPU0,GPU1)` / `(GPU2,GPU3)` paired placement:

| mode | pp64 | tg512 |
|---|---:|---:|
| two-GPU layer split | 1112.31 t/s | 202.92 t/s |
| two-GPU EP, NCCL | 1180.42 t/s (`+6.12%`) | 171.44 t/s (`-15.52%`) |
| two-GPU EP, internal P2P AR | 1219.12 t/s (`+9.60%`) | 178.66 t/s (`-11.95%`) |

The specialized internal two-GPU AllReduce helped, but paired EP still did not beat layer-split decode. The more invasive paired-weight placement was therefore not implemented.

## Decision

Keep `DSV4_EXPERT_PARALLEL=1` as an opt-in prefill/batch mode, not the interactive production default. On the full 284B model it raises pp512 throughput by about 20.9%, but reduces decode throughput by about 34.4%.

This mode is useful for prompt ingestion, offline batching, embeddings-style workloads, or a dedicated prefill worker. It is not useful for a single-process interactive server that must use the same weight placement for prefill and decode.

The next useful optimization would be communication/computation overlap or token dispatch that avoids broadcasting every hidden vector to every expert shard. More NCCL protocol tuning, FP32 AllReduce, and a manual reduce-to-one path were tested and were slower than default NCCL `RING_LL`.
