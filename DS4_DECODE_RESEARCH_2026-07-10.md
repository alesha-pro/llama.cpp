# DeepSeek-V4 decode optimization audit — 2026-07-10

## Baseline and scope

Hardware: 4 x RTX 3090 at 220 W, custom 610.43.02 P2P driver, `GGML_CUDA_P2P=1`.

Production model:

`/mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

All tests used the existing DS4 optimizations including the fused Lightning Indexer, device-resident MoE scheduler, fused prefill up/gate path, layer split, and no MTP. The full-model shallow-context layer baseline was consistently about 35.7-36.3 t/s. The previously verified long-context result remains 31.2-31.6 t/s around 97K tokens.

## Hypotheses tested in this pass

### One scheduler copy instead of pipeline copies

Disabling pipeline scheduling reduced `sched copies` from 4 to 1. On the four-layer fixture this looked spectacular: 177.00 -> 229.35 t/s (`+29.58%`). It did not transfer to the 43-layer model:

| mode | full-model tg256 mean |
|---|---:|
| normal pipeline scheduler | 35.7172 t/s |
| one scheduler copy | 35.6764 t/s |
| delta | `-0.11%` |

It also reduced fixture pp4096 prefill by about 3.1% and a real 10,840-token prompt by about 5.2%. The experimental flag was removed.

Conclusion: scheduler overhead dominates an artificially shallow four-layer graph, but is amortized by the real 43-layer model.

### Decode fused up + gate MMVQ

The existing prefill fusion was extended experimentally to the one-token MMVQ path. Three-run fixture means changed from 172.86 to 164.46 t/s (`-4.86%`). Register pressure/occupancy outweighed the saved launch and activation read. The decode extension was removed; the successful prefill fusion remains.

### Replace six views + five adds with one reduction

A strided CUDA sum-rows path reduced the four-layer decode graph from 508 to 476 nodes. One-token logits were bit-identical (`SHA-256 0b1872ebc520342f26e4864106bdeba8a80268a7ba4ff4fd33aab45267f20102`).

Fixture decode improved by about 1.6%, but full-model A/B was flat:

| mode | full-model tg256 mean |
|---|---:|
| chained adds | 35.6989 t/s |
| strided reduction | 35.6880 t/s |
| delta | `-0.03%` |

The experiment was removed.

Conclusion: the five tiny add kernels are not on the meaningful full-model critical path.

### Fused routed down projection + top-6 accumulation

A Q2_K decode kernel computed all six routed down projections inside one CTA and wrote a compact hidden vector. Three launch geometries were tested on the real-weight fixture:

| geometry | result versus same-build layer baseline |
|---|---:|
| one warp/expert, two rows/block | about `+0.78%` in one pair; too close to noise |
| two warps/expert | `-2.28%` |
| four warps/expert | `-3.36%` |

The larger CTAs reduced block count but hurt scheduling/occupancy. The entire experiment was removed.

### Four-GPU row/tensor split

The stock row-split path cannot currently run DeepSeek-V4. `attn_output_a.weight` becomes a CUDA split buffer, then grouped attention applies `RESHAPE -> MUL_MAT_ID`; CUDA `MUL_MAT_ID` explicitly does not support split buffers. The graph is rejected during reserve. Supporting this would require a broad split-buffer implementation for grouped attention and routed experts, with frequent per-layer synchronization, so it was not pursued.

### Expert parallel decode

Full-model four-way EP gives a real pp512 prefill win of about `+20.86%`, but tg128 decode falls from 36.34 to 23.82 t/s (`-34.44%`). Two-way EP and the specialized internal P2P AllReduce reduce the penalty but still remain about 12% below layer split on the fixture. See `DS4_EXPERT_PARALLEL_2026-07-10.md`.

## Bottom line

The tested graph/scheduler micro-overheads are not the production decode bottleneck. The remaining short-context critical path is dominated by routed-expert MMVQ weight reads and the rest of the per-layer compute; at long context the fused Lightning Indexer already removed the prior index-cache collapse.

The clean build retains only proven changes. No experimental decode flags from this audit remain in source. The most valuable verified new mode is `DSV4_EXPERT_PARALLEL=1` for prefill-heavy workloads, not interactive decode.
