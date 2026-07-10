# DS4 raw decode: 128K Top-K research and exact radix-select

Date: 2026-07-10

Goal: raise raw (no-MTP) decode toward 35 tok/s at 128K-256K without a quality shortcut.

## What the 128K trace showed

`llama-ds4-topk-overlap` captures the masked Lightning Indexer scores and exact Top-512 from all 21 CSA layers. The callback intentionally synchronizes the graph, so its timings are not performance measurements. The reusable 16-step trace is `/tmp/ds4-topk-128k.bin` (129,960-131,040-token region, depending on the run).

At 131,040 prompt tokens:

- consecutive exact Top-512 overlap: mean 0.608563, minimum 0.146484;
- adjacent-CSA-layer overlap: mean 0.251416;
- naive half-cache TISA: mean 0.748878, minimum 0.373047;
- K2 plus a 4096-row adaptive band: overlap 0.985534/0.828125 mean/min, router-score mass 0.996548/0.927092;
- K4 plus a 12,288-row adaptive band: overlap 0.996478/0.886719, score mass 0.999088/0.963667;
- the same idea at the WMMA 32-row block granularity was not safe enough: K4/hot1024 refreshed 52.75% of blocks but reached only 0.933011/0.689453 overlap.

Conclusion: naive temporal reuse is rejected. Scatter hot-band reuse looks good offline but does not translate into fewer WMMA cache reads because the selected rows touch almost every 32-row tile.

## Exact Top-512 work

The DS4 gather decode path used `ggml_argsort_top_k`, which performs a complete radix argsort and then views its first 512 indices. At the real 128K index width (`N=32768`, one row):

- full CUDA argsort: 131.05 us;
- CUB DeviceTopK: 134.58 us (rejected: 2.7% slower);
- first single-block radix-select: 136.73 us (rejected);
- multi-launch radix-select: 84.79 us;
- final cooperative radix-select: 35.44 us, 3.70x faster than full argsort.

The final kernel performs four device-side byte-radix cutoff passes and compacts exactly 512 indices. It includes a validated non-cooperative fallback. CUDA correctness tests pass for unique values and cutoff ties.

Enable in DS4 with:

```bash
DSV4_DECODE_RADIX_TOPK=1
```

The returned set is exact but unsorted. Gather attention is mathematically order-invariant; floating-point accumulation order can still change a later greedy token when logits are nearly tied.

## Full server A/B

Identical server settings: no MTP, 4x3090, layer split, `-ts 1,1,1,1`, compact SWA, F16 cache, `ubatch=512`, P2P, resident MoE and all proven fusions. Identical 390,000-byte source prompt tokenized to 129,960 tokens, greedy generation, 64 output tokens.

| Variant | ms/token | raw tok/s |
|---|---:|---:|
| Full argsort baseline | 30.6811 | 32.5934 |
| Exact radix-select | 30.2543 | 33.0532 |
| Exact radix-select, warmed cached sample | 29.5823 | 33.8040 |

Cold matched A/B gain: +1.41%. A longer warmed 128-token sample was 32.495 tok/s, so short-run variance is material; the conservative accepted result is the matched cold A/B.

Control at shallow context on the same optimized build/split: 595 prompt + 128 generated = 35.699 tok/s. Therefore the general decode path has not regressed; the remaining long-context gap is the Lightning Indexer path.

## Rejected side paths

- Q8_0 index cache: fused indexer 53.97 -> 52.53 us at N=32768, only +2.7%; Q4_0 was slower at 54.34 us.
- Raising GPU power limit: read-only sampling showed roughly 140-173 W/GPU during decode under the existing 220 W limit, so the cards were not power-capped. Memory stayed at 9501 MHz.
- Simple device TopK, naive TISA, and WMMA-block HISA: insufficient speed or quality as detailed above.

## Multi-GPU sharding feasibility check

Real decode-shaped microbenchmarks changed the initial sharding estimate:

- Lightning Indexer, F16 K, one query, 8,192 rows: 36.91 us;
- Lightning Indexer, F16 K, one query, 32,768 rows: 55.27 us;
- exact radix Top-512, 8,192 scores: 33.65 us;
- exact radix Top-512, 32,768 scores: 34.82 us.

The Top-512 kernel is launch/synchronization-bound at these widths. A conventional four-GPU pipeline -- four local 8K scans, four local Top-512 selections, then a Top-512 merge over 2,048 candidates -- is therefore about 36.9 + 33.7 + 33 us before P2P orchestration, versus about 55.3 + 34.8 us locally. It would be slower, not faster.

The full 32K Lightning scan is only about 1.16 ms/token across all 21 CSA layers. Even an impossible zero-overhead 4x acceleration saves about 0.87 ms/token, moving the matched 30.25 ms/token result only to roughly 29.38 ms/token (about 34.0 tok/s). Sharding the scan alone cannot reach a stable 35 tok/s.

Nsight Compute 2025.1 was available, but hardware-counter collection is disabled for the current user (`ERR_NVGPUCTRPERM`). Timing-based conclusions do not depend on those counters.

## Remaining exact kernel hypothesis

A sharded implementation is worthwhile only if it fuses scoring and local selection into a persistent cooperative kernel, keeps the cache replicated as it grows, and merges candidates without a separate general Top-K launch. Even then, the realistic whole-token gain is on the order of 0.4-0.7 ms rather than multiple milliseconds.

For a substantial quality-preserving jump beyond that ceiling, the algorithmic path is speculative decoding (a small external draft or a self-speculative shallow draft) with exact target-model verification. It preserves the target distribution; unlike MTP it does not require the model's MTP head, but its acceptance rate must be measured on this exact DS4 quant and workload.

## Original sharded design sketch

At the same build, shallow decode is 35.70 tok/s and 126.9K decode is about 33 tok/s. Local Top-K work recovered only part of the 2.2-2.7 ms length tax. Reaching 35 without approximation requires removing roughly three quarters of that tax.

The next exact design is a P2P-sharded Lightning Indexer:

1. shard or replicate each CSA index cache across four GPUs as it grows (one 128xF16 row every four tokens);
2. broadcast the small per-layer query and 64 router weights;
3. run four concurrent local score scans and local exact Top-512 selections;
4. merge at most 2048 candidates on the owning layer GPU;
5. keep all orchestration device-resident and CUDA-graph compatible.

This design remains valid as an experiment only in fused form. The measured component timings reject the unfused implementation and show that sharding alone does not have enough theoretical leverage for the 35 tok/s target.
