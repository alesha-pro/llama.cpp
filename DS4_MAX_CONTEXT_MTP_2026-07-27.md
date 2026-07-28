# DS4 284B: maximum context with MTP on 4x RTX 3090

Date: 2026-07-27. Follow-up to `DS4_PREFILL_RADIX_TOPK_2026-07-27.md`, which
made MTP + long context possible at all. Question here: how far does it go, and
at what speed.

All runs use the full 284B checkpoint from NVMe, `llama-server`, MTP enabled,
and the current flag set including `DSV4_MMVQ_SMALLK=1` and
`DSV4_PREFILL_RADIX_TOPK=1`.

## Two separate ceilings

The prefill argsort ceiling (fixed by `DSV4_PREFILL_RADIX_TOPK`) was not the
only one. Testing deeper prompts exposed a second, independent limit:

| depth | prefill | decode |
|---:|---|---|
| 104,868 | completes | completes, 41.624 t/s |
| 114,086 | **completes** | **OOM** |

At 114K the failure is not in prefill at all:

```
dsv4-mtp: state shadows ready (43 layers)
dsv4: DSV4_DECODE_FUSED_IDX=1 - using fused decode Lightning Indexer
ggml-cuda.cu:108: CUDA error
CUDA error: out of memory
  current device: 2, in function ggml_cuda_graph_evaluate_and_capture at ggml-cuda.cu:4581
  cudaGraphLaunch(graph->instance, cuda_ctx->stream())
```

Prefill reached `progress = 0.999965` and then the **MTP decode CUDA graph could
not be instantiated**. So the limit is decode-graph memory, and it lands on the
device with the least headroom.

## The real constraint is the tensor split, not the total

Memory at `--ctx-size 131072` with MTP, production `-ts 1,1,1,0.85`:

| GPU | free | model | context | compute | unaccounted |
|---|---:|---:|---:|---:|---:|
| CUDA0 | 480 | 22617 | 219 | 409 | 397 |
| CUDA1 | 1894 | 20779 | 258 | 787 | 403 |
| CUDA2 | **10** | 22652 | 261 | 793 | 405 |
| CUDA3 | 2452 | 15638 | 174 | 822 | 5035 (MTP) |

Totals: 81,686 MiB of model + ~5,035 MiB of MTP weights + 912 MiB of KV +
2,811 MiB of compute in 96,492 MiB of VRAM. About 4.8 GiB is free **in
aggregate** — but 10 MiB of it is on CUDA2, which is where the decode graph
needs room.

`-ts 1,1,1,0.85` skews load onto GPUs 0-2 specifically to free GPU3 for the MTP
weights. With MTP now viable at depth, that skew is counterproductive: it starves
the exact device that has to host the deepest decode graph.

### Rebalancing

Layer granularity is coarse — 43 layers over 4 GPUs is about 1,900 MiB per
layer — so the useful move is to shift exactly one layer off CUDA2. `-ts
1,1,0.90,0.95` does that:

| GPU | model before | model after | free before | free after |
|---|---:|---:|---:|---:|
| CUDA0 | 22617 | 22617 | 480 | **684** |
| CUDA1 | 20779 | 20779 | 1894 | **2260** |
| CUDA2 | 22652 | **20750** | **10** | **2312** |
| CUDA3 | 15638 | **17540** | 2452 | **774** |

Minimum free across devices goes from **10 MiB to 684 MiB**, and the total model
footprint is unchanged (81,686 MiB either way) — this is purely redistribution.
The compute buffers also came out smaller (286-635 vs 409-822 MiB), which is a
second-order effect of the layer redistribution.

## Results

All with MTP enabled, `llama-server`, `ubatch 512`, `--parallel 1`:

| split | ctx | prompt | prefill | decode 64 | decode 128 warm | outcome |
|---|---:|---:|---:|---:|---:|---|
| `1,1,1,0.85` | 131072 | 104,868 | 414.20 | 41.035 | 41.624 | ok |
| `1,1,1,0.85` | 131072 | 114,086 | completed | — | — | **decode-graph OOM** |
| **`1,1,0.90,0.95`** | **131072** | **127,356** | **395.93** | 35.528 | **42.648** | **ok, context full** |

So the rebalanced split takes the usable depth from "fails somewhere between
105K and 114K" to **the full 131,072-token context**, and decode at that depth
is *faster* than the earlier 105K run (42.648 vs 41.624 t/s) — the extra
headroom is not costing anything.

The `decode 64` column is a cold first response and includes graph capture; the
`decode 128 warm` column is the steady rate and is the number to quote.

### 262144 does not fit

Same rebalanced split, `--ctx-size 262144`, ~242K-token prompt. The KV cache
allocated fine and prefill ran **past 131072 without trouble**, then failed at
204,800 tokens (84.5%):

```
slot update_slots: prompt processing progress, n_tokens = 204800, progress = 0.844916
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1189.40 MiB on device 3:
    cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA3 buffer of size 1247175168
```

This is a third distinct limit: a **graph-reserve compute buffer that scales
with depth**. At 205K it wanted 1,189 MiB on CUDA3, which holds the ~5 GiB of
MTP weights and had 774 MiB free.

The scaling explains why 131072 is the boundary rather than a coincidence: the
buffer grows roughly linearly with the compressed width, so
`774 / 1189 x 205K` is about **133K** — essentially exactly the 131,072 that
does work. The configuration is at its edge, not comfortably inside it.

Total VRAM is not the binding constraint (6,030 MiB free in aggregate at
131072, versus roughly 2,400 MiB of extra demand at 262144). It is again
distribution: moving one more layer off CUDA3 would give it the headroom, but
the receiving device then comes up ~100 MiB short once its own KV and compute
growth are counted. There is no layer-granular split that satisfies both.

**Untested lever for going further:** `--ubatch-size 256` would roughly halve
the compute buffers (they are 286-635 MiB at ubatch 512), freeing on the order
of 1.5 GiB in aggregate. That may be enough for 262144, at some prefill cost.
Not attempted here.

### Extrapolating between 131072 and 262144

CUDA3 is the binding device because it carries the ~4,957 MiB of MTP weights:

```
CUDA3:  24123 total - 17540 model - 4957 MTP = 1626 MiB for KV + compute
```

KV on CUDA3 is 215 MiB at ctx 131072, i.e. **0.00164 MiB/token**. For the
compute buffer there is only one deep measurement (1,189.40 MiB at depth
204,800), so it was extrapolated two ways to bound the answer:

- **A**, purely proportional: `compute = 0.005808 * D`
- **B**, affine off the 635 MiB load-time value: `compute = 635 + 0.002707 * D`

Requiring `compute(C) + KV(C) <= 1626` to fill a context of size `C`:

| ctx | KV | available | need (A) | margin A | need (B) | margin B |
|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 215 | 1411 | 761 | +650 | 990 | +421 |
| 147456 | 242 | 1384 | 856 | +528 | 1034 | +350 |
| **163840** | 269 | 1357 | 952 | **+405** | 1078 | **+279** |
| 174080 | 286 | 1340 | 1011 | +329 | 1106 | +234 |
| 196608 | 322 | 1304 | 1142 | +162 | 1167 | +137 |
| 204800 | 336 | 1290 | 1189 | +101 | 1189 | +101 |
| 262144 | 430 | 1196 | 1523 | **-327** | 1345 | **-149** |

The model reproduces the observed failure rather than being fitted to it after
the fact: at ctx 262144 the available budget was `1626 - 430 = 1196` MiB against
a 1,189.40 MiB request — a **+7 MiB** margin, and it failed. That also puts a
floor on the allocator overhead (old buffer not released before the new one, or
pool fragmentation): `epsilon >= 7 MiB`.

Predicted fill limit: **~205-220K**, or ~200-210K allowing for fragmentation.

### 163840 measured — fits, as predicted

Same split and flags, `--ctx-size 163840`, 156,423-token prompt:

| ctx | prompt | prefill | decode 64 | decode 128 warm | outcome |
|---:|---:|---:|---:|---:|---|
| 131072 | 127,356 | 395.93 | 35.528 | 42.648 | ok |
| **163840** | **156,423** | **367.19** | 37.948 | **41.386** | **ok, context full** |
| 262144 | — | fails at 204,800 | — | — | compute-buffer OOM |

The prediction held: 163840 was projected to have 279-405 MiB of margin on CUDA3
and it completed with no allocation failure. That is now a second confirmation
of the model (131072 works, 163840 works, 262144 fails where predicted).

Cost of the extra 29K tokens of depth:

- prefill **395.93 -> 367.19 t/s (-7.3%)**;
- decode **42.648 -> 41.386 t/s (-3.0%)**.

Both degrade gently and roughly in line with the indexer/attention share
growing with depth. Ingesting the full 156K prompt takes about 7.1 minutes.

**So the practical maximum is 163840 at 41.4 t/s decode**, with 174080 (~170K)
also projected to fit at a similar margin. 196608 remains the untested boundary
case (projected margin ~140-160 MiB, versus the +7 MiB that failed).

### Where this sits historically

| configuration | context reached | decode |
|---|---:|---:|
| recorded reference, no MTP (`DS4HANDOFF.md` section 6) | ~97K | 31.2-31.6 |
| equal split, no MTP, `+SMALLK` | ~105K | 34.8 |
| `1,1,1,0.85` + MTP + `SMALLK` + `PREFILL_RADIX_TOPK` | ~105K | 41.6 |
| **`1,1,0.90,0.95`, same flags** | **127K (full 131072)** | **42.6** |

## Answer

**The maximum verified context with MTP on the 284B checkpoint is
`--ctx-size 163840`, and it fills completely** — 156,423 tokens at
**367.19 t/s prefill and 41.386 t/s warm decode**. 131072 is the faster point
(395.93 / 42.648) if 128K is enough. 262144 fails at ~205K depth on a
compute-buffer allocation; the projected ceiling is ~205-220K.

Required launch changes versus `DS4HANDOFF.md` section 4:

```bash
-ts 1,1,0.90,0.95          # not 1,1,1,0.85 - see above
DSV4_PREFILL_RADIX_TOPK=1  # without it, prefill dies at 90K
DSV4_MMVQ_SMALLK=1         # +5% decode
```

If more than 128K is needed, the K144 REAP 162B checkpoint already runs 260K at
~14.8 GiB/GPU (`DS4_REAP_K144_2026-07-21.md`) — it has 9 GiB/GPU more headroom
and does not hit any of these ceilings. It has no MTP weights, so it trades the
+20% speculation gain for the context.
