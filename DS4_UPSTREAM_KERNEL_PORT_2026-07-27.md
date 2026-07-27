# DS4 fork vs current upstream llama.cpp — kernel audit and ports

Date: 2026-07-27. Goal for this pass: a **balanced** configuration for a single
interactive agentic user (large initial prefill, then many rounds of decode plus
small tool-result prefills), not a single-metric win.

Upstream reference: `/mnt/nvme/engines/llama.cpp` fast-forwarded to
`0e4a03622` (2026-07-27). Fork base is `781e978f3` (2026-05-14), so upstream is
about 2.5 months and 141 commits ahead on the shared tree.

## 0. Why decode is the target for the agentic profile

Session model for a coding agent: one large context ingest, then N rounds of
(decode a response, prefill a tool result).

At the current measured rates, a 50-round session over a 130K context spends
roughly:

- initial prefill: 130,000 / 500 t/s = 260 s;
- incremental prefills: 50 x 2,000 / 500 t/s = 200 s;
- decode: 50 x 500 tokens / 31 t/s = 806 s.

Decode is about **64%** of wall clock and is the correct optimization target,
with the constraint that prefill must not regress.

## 1. Architectural comparison with upstream

Upstream has since implemented several DSV4 pieces the fork already had. The
important finding is that on the long-context path the fork is **not** behind:

| area | upstream (0e4a03622) | this fork |
|---|---|---|
| indexer top-k applied to attention | `build_top_k_mask`: `-INF` fill + `set_rows` zeros, then **dense FA over the whole KV cache** (O(n_kv) compute) | physical gather of only the selected 512 rows (O(top_k) compute) |
| Top-512 selection | `ggml_top_k` (CUB / bitonic argsort) | exact cooperative radix-select, 35.4 us vs 131 us at N=32768 |
| Lightning Indexer CUDA op | `lightning-indexer.cu`, 588 lines, vec + wmma | 930 lines: adds q-tiled, fp16-acc and causal-skip variants |
| hyper-connection ops | `dsv4-hc.cu` (`hc_comb`/`hc_pre`/`hc_post`) | `dsv4-hc-expand` / `-split-sinkhorn` / `-weighted-sum` |
| MoE tiling | worst-case `ncols_max` | real per-expert bound + device-resident scheduler |
| expert parallel / EP-to-layer | none | opt-in, one-shot device relayout |

Upstream's dense-FA-with-mask approach means its attention cost still grows with
full context length; the fork's gather keeps it at the top-k width. This is the
main reason the fork's long-context numbers are not reachable upstream, and it
is why a wholesale merge is not attractive.

Upstream also **removed `-sm row`** entirely (#24216), independently confirming
the fork's earlier finding that row split is not viable for DSV4, and it
**reverted** its own attempt to reduce scheduler synchronizations (#20793,
reverted by 86b94708f) — so no ready-made fix exists upstream for the
host-sync overhead documented in `DS4_DECODE_UPLIFT_RESEARCH_2026-07-10.md`.

## 2. Hypotheses

### H1 — MMVQ accumulators spill to local memory (upstream 683f0c72e). ACCEPTED

Upstream commit `683f0c72e` ("Only index by compile times + always
multiply/add", NVIDIA) restructures the write-back of `mul_mat_vec_q`: the final
result was read as `tmp[j][threadIdx.x]`, a **runtime** index into a per-thread
array, which forces the whole accumulator array into local memory. Moving the
write inside the unrolled `i` loop and indexing by the compile-time `i` keeps it
in registers. Biases are zero-initialized, so the `use_bias` / `use_gate_bias`
branches are replaced by unconditional adds.

This matters here because at decode the routed experts run through
`mul_mat_vec_q<type, ncols_dst=1, has_fusion=true, small_k>` — the fork's
hottest decode kernel.

**Static verification before any benchmark** (`nvcc -Xptxas -v`, sm_86, same
flags as the real build, 264 kernels in the TU):

| kernel (ncols_dst=1) | before | after |
|---|---|---|
| **IQ2_XXS, fusion, small_k** (routed up/gate) | 98 regs, **32 B stack frame** | 96 regs, **0 B** |
| IQ2_XXS, fusion | 51 regs, 0 B | 51 regs, 0 B |
| Q8_0, fusion, small_k | 64 regs, 0 B | 64 regs, 0 B |
| Q2_K, fusion, small_k | 89 regs, 0 B | 89 regs, 0 B |
| whole TU | max **64 B** stack frame | max **0 B** |

The 32-byte spill is exactly `float tmp[1][4]` plus `float tmp_gate[1][4]` for
`rows_per_cuda_block = nwarps = 4`, i.e. both accumulators were living in local
memory and every `tmp[j][i] +=` in the inner K loop was a local load/store.
Across the translation unit, 159 of 264 kernels changed and **every**
local-memory spill is gone. Two non-production shapes gained registers (Q2_K
fusion non-small_k 48 -> 54, Q6_K fusion 40 -> 48) with no stack frame in either
version, which is harmless at this occupancy.

**Correction — which variant DSV4 actually runs.** Working out the dispatch
rather than assuming it: at decode `ncols_dst = ids ? ne2 : ne1` and for
`MUL_MAT_ID` with dst `[n_ff_exp, n_expert_used, n_tokens]` that is
`ne2 = n_tokens = 1`, so the routed experts do use
`mul_mat_vec_q<type, 1, has_fusion, small_k>`. But `should_use_small_k` gives,
for the routed up/gate matmul (IQ2_XXS, `ncols_x = n_embd = 4096`,
`qk = 256`, `qi = QI2_XXS = 16`, `vdr = 2`, `nwarps = 4`):

```
blocks_per_row_x      = 4096 / 256      = 16
blocks_per_iter_1warp = 2 * 32 / 16     = 4
threshold             = nwarps * 4      = 16     ->  16 < 16 is FALSE
```

So DSV4 runs the **`small_k = false`** variant, which had **no spill in either
version** (51 registers, 0 B, unchanged). `ffn_down` (Q2_K, `ncols_x = 2048`,
`vdr = 1`, `qi = QI2_K = 16`) works out to `blocks_per_row_x = 8` against a
threshold of `4 * (1*32/16) = 8`, so it too fails the strict `<` and also runs
`small_k = false`. Neither production shape was spilling.

Therefore **H1 is expected to be performance-neutral for DSV4 specifically**.
It is kept anyway: it is a faithful upstream port, statically strictly better or
equal on every kernel in the TU, and it is a prerequisite for testing H5 below
on its merits. It must not be reported as a DSV4 speedup.

Measured effect: see section 3.

### H5 — DSV4's routed up/gate sits exactly on the `small_k` boundary. TESTED

Falling out of the H1 analysis: `blocks_per_row_x == nwarps *
blocks_per_iter_1warp` exactly. And it is not only the up/gate matmul — working
the boundary out for every production tensor type shows that **both** routed
expert projections sit on it, while attention does not:

| tensor | quant | ncols_x | blocks/row | threshold | `<` | `<=` |
|---|---|---:|---:|---:|---|---|
| routed up/gate | IQ2_XXS | 4096 | 16 | 16 | false | **true** |
| routed down | Q2_K | 2048 | 8 | 8 | false | **true** |
| attention / shared / output | Q8_0 | 4096 | 128 | 32 | false | false |

This is not really a coincidence: `n_embd = 4096` and `n_ff_exp = 2048` with
these two 2-bit quants both land the reduction dimension exactly at what a
4-warp block covers in one iteration.

The upstream comment describes the intent as "trigger when the full thread block
covers all K blocks in a single loop iteration", which is precisely what
equality means, yet the strict `<` excludes it. At equality the non-small_k
variant runs a single iteration of MAC work and then pays a full cross-warp
shared-memory reduction plus `__syncthreads` for it. The `small_k` variant gives
each block `nwarps` output rows instead of one, amortizing that reduction over
4x more MAC work, at the cost of 4x fewer blocks (less parallelism across SMs).
Which side wins is not decidable on paper.

Because this affects only the `ncols_dst == 1` MMVQ path, it is a **decode-only**
change: prefill goes through MMQ and is untouched by construction.

Added as `DSV4_MMVQ_SMALLK=1`, which relaxes `<` to `<=`. Both template
instantiations already exist and the choice is host-side, so this costs nothing
when the flag is unset.

The flag is shape-driven, not expert-count-driven, so it should apply equally to
the K144 REAP 162B checkpoint: REAP prunes the number of experts, not their
width, and `DS4_REAP_K144_2026-07-21.md` records that both models "still execute
six same-width experts per token". Same `n_embd` and `n_ff_exp` means the same
boundary. Not yet measured on that checkpoint.

Note the interaction with H1: the variant this flag switches DSV4 onto
(`IQ2_XXS, ncols_dst=1, fusion, small_k`) is precisely the one that carried the
32-byte local-memory spill before H1. So H1 is a prerequisite for measuring H5
on its merits. Whether the spill would actually have eaten the whole H5 gain is
measured separately in section 3.3 rather than assumed.

Result: see section 3.

### H2 — MoE activation quantization is done n_expert_used times (upstream #25441). REJECTED BY ARITHMETIC

Upstream `5839ba352` notes that for MoE gate/up the activation is broadcast
across experts (`ne11 == 1`), so every token's row is re-quantized once per
routed slot; it adds `quantize_scatter_mmq_q8_1_cuda` to quantize once and
scatter. The fork has the same redundancy: `mmq.cu` quantizes
`ne_get_rows = ne12 * n_expert_used` rows.

The saving was computed before implementing it:

- per MoE layer per ubatch of 512: read drops from 512 x 8 x 4096 floats
  (64 MB) to 8 MB; the write side is unchanged;
- about 40 routed layers, so about 2.24 GB avoided per ubatch, or about 560 MB
  per GPU under the layer split;
- at roughly 750 GB/s that is **about 0.75 ms**;
- one 512-token ubatch at 495 t/s takes about **1034 ms**.

That is **0.07%** of prefill, i.e. far below run-to-run noise. The `down`
projection cannot use this at all, because each expert slot has its own
intermediate activation (`ne11 != 1`).

The deeper reason it cannot pay off: DSV4 prefill on this rig is **compute
bound, not bandwidth bound**. One ubatch touches essentially the whole expert
weight set (about 80 GB across four GPUs, about 107 ms of pure DRAM time) but
actually takes about 1034 ms, so the MMQ int8/dequant ALU work dominates. A
bandwidth-side optimization of this size cannot show up. Not implemented.

### H3 — vectorized `get_rows` (upstream #25929 / #25962). REJECTED BY ARITHMETIC

`k_get_rows_float` copies one element per thread; upstream adds an int4 (16 B)
vectorized path for the contiguous same-type case. The fork's decode gather
(`ggml_get_rows(comp_rows, topk)`) moves 512 rows x ~576 values x 4 B = about
1.2 MB per CSA layer per token, about 25 MB per token over 21 layers, i.e. about
**0.03 ms** against a ~32 ms token. Below noise, and the DS4 gather is a
F16-to-F32 cast, which upstream's vectorized path explicitly excludes. Not
implemented.

### H4 — MTP speculative decode has never been measured at long context. TESTED

Every long-context number on record (`DS4HANDOFF.md` section 6,
`DS4_EP_TO_LAYER`, `DS4_REAP_K144`) is explicitly **no MTP**, and the only MTP
numbers are short-context. `DS4_DECODE_UPLIFT_RESEARCH_2026-07-10.md` lists this
as open hypothesis A1 with zero code required. For the agentic profile this is
the single largest potential lever: 31 t/s x an accepted 1.2-1.5 draft rate
would be 37-46 t/s.

The answer turned out not to be a throughput number at all: the combination is
**blocked by VRAM before it can be measured**. MTP requires `-ts 1,1,1,0.85`,
and that split is exactly what makes long-context prefill run out of memory. See
section 3.3. This is why the hypothesis had stayed open since 2026-07-10 — it is
not that nobody ran it, it is that it cannot run as configured.

Result: see sections 3.3 and 3.4.

### H6 — server context checkpoints tax long prefill. REFUTED

Discovered while running the H4 long-context arms: `llama-server` prefill of a
~105K-token prompt was visibly far slower than the same token count through
`llama-batched-bench`, and the server log showed it creating a context
checkpoint every 8192 tokens:

```
slot create_check: id 0 | task 0 | created context checkpoint 8 of 32
                   (pos_min = 65535, pos_max = 65535, n_tokens = 65536, size = 482.762 MiB)
```

At 482.8 MiB per checkpoint and the default `--checkpoint-every-n-tokens 8192`,
a 130K prompt writes about **16 checkpoints, roughly 7.7 GiB of copies**, none
of which `batched-bench` performs.

Checkpoints exist so a slot can rewind to a mid-prompt state. A single-user
agentic session is **append-only**: each turn extends the context and the server
reuses the whole cached prefix, so a mid-prompt rewind point is rarely used. MTP
does not need them either — `6e8258c` already gave `dsv4-mtp` a custom rollback
and bypasses the per-round spec checkpoints.

The checkpoints do **grow with context** — observed sizes 482.762, 536.512 and
590.262 MiB for checkpoints 8, 9 and 10, a difference of about 53.75 MiB per
8192 tokens, i.e. about **6.7 KiB/token**, exactly the MLA compressed-KV rate
recorded in `DS4HANDOFF.md`. So the total is quadratic in context length,
roughly 7 GiB at 130K.

**This hypothesis was wrong.** Disabling checkpoints did not change anything —
see section 3.3. Recorded here because the reasoning looked sound and the
disconfirming test is the useful part.

## 3. Measurements

All runs: production 284B GGUF from the NVMe copy, 4x RTX 3090 at 220 W, layer
split `-ts 1,1,1,1`, `ubatch 512`, FlashAttention, P2P, no MTP, ctx 16384, and
the full production flag set (`DSV4_CONSTANT_SHAPE`, `SPARSE_FA`, `FA_UNION`,
`IDX_SKIP`, `MOE_TILE`, `MOE_RESIDENT`, `GLU_FUSE`, `MOE_FUSE`,
`DECODE_FUSED_IDX`, `DECODE_RADIX_TOPK`). `llama-batched-bench`, same binary for
the H1 and H5 arms — H5 differs only by the `DSV4_MMVQ_SMALLK=1` environment
variable, so the two arms are not even separated by a rebuild.

Decode throughput, `tg = 256`, three runs per arm, arms interleaved in time so
any thermal or clock drift is shared:

| arm | n | pp512 decode (min-max) | pp8192 decode (min-max) |
|---|---:|---:|---:|
| pre-port baseline | 1 | 35.461 | 33.982 |
| H1 (port applied) | 3 | **36.009** (35.735-36.188) | **33.886** (33.815-34.000) |
| H5 (`DSV4_MMVQ_SMALLK=1`) | 3 | **37.971** (37.913-38.044) | **35.549** (35.489-35.582) |

Prefill over the same runs:

| arm | pp512 prefill | pp8192 prefill |
|---|---:|---:|
| baseline | 559.38 | 550.25 |
| H1 | 578.36 | 556.57 |
| H5 | 582.29 | 555.80 |

### 3.1 H1 (MMVQ register indexing) — neutral, as predicted

H1 moves decode by +1.5% at pp512 and -0.3% at pp8192, i.e. it is **within noise
for DSV4** — exactly what the dispatch analysis predicted, since DSV4 does not
run the variant that was spilling. It is kept for the reason given above, and is
**not** claimed as a speedup.

Correctness: `test-backend-ops -b CUDA0 -o MUL_MAT_ID` 764/764 pass, `-o
MUL_MAT` pass.

### 3.2 H5 (`DSV4_MMVQ_SMALLK=1`) — real decode win, prefill untouched

| shape | H1 mean | H5 mean | delta |
|---|---:|---:|---:|
| pp512 + tg256 | 36.009 | 37.971 | **+5.45%** |
| pp8192 + tg256 | 33.886 | 35.549 | **+4.91%** |

The signal is far outside the noise floor: the three H5 repeats span 0.13 and
0.09 t/s, the three H1 repeats span 0.45 and 0.19 t/s, and the two arms' ranges
**do not overlap** — the gap is 1.96 and 1.66 t/s, roughly an order of magnitude
larger than the spread within either arm.

Prefill is unchanged (582.29 vs 578.36, and 555.80 vs 556.57 — both within the
prefill noise band), which is expected by construction: the flag only reaches the
`ncols_dst == 1` MMVQ path, and prefill runs through MMQ.

This is the balanced outcome the pass was aiming for: decode up ~5%, prefill
flat.

### 3.3 The real long-context ceiling: `-ts 1,1,1,0.85` OOMs at ~90K

Attempting the H4 long-context arms through `llama-server` with **default**
settings (`--checkpoint-every-n-tokens 8192`, `--ctx-checkpoints 32`,
`--ctx-size 131072`, `-ts 1,1,1,0.85`, production flags) did not merely run
slowly — it **crashed**:

```
slot create_check: created context checkpoint 10 of 32 (n_tokens = 81920, size = 590.262 MiB)
slot update_slots: prompt processing progress, n_tokens = 90112, progress = 0.859290
ggml/src/ggml-cuda/ggml-cuda.cu:108: CUDA error
CUDA error: out of memory
  current device: 2, in function alloc at ggml-cuda.cu:533
  cuMemCreate(&handle, reserve_size, &prop, 0)
  ... argsort_f32_i32_cuda_cub -> ggml_cuda_op_argsort
```

The 105K-token prompt died at about **86% of prefill (~90K tokens)** when the
prefill top-k `argsort` could no longer reserve its pool on CUDA2 — the device
that carries 22.65 GiB of model under `-ts 1,1,1,0.85`. The growing checkpoints
were the obvious suspect (H6), so that was tested first.

Re-running the identical arm with `--checkpoint-every-n-tokens -1` (verified: no
`create_check` lines at all) **failed at exactly the same point** — `n_tokens =
90112, progress = 0.859290`, same `argsort_f32_i32_cuda_cub` ->
`ggml_cuda_pool_vmm::alloc` -> `cuMemCreate` OOM on CUDA2. So **H6 is refuted**:
context checkpoints are not the cause and evidently do not live in the VRAM that
runs out (`--cache-ram` defaults to 8192 MiB of host RAM, which fits them).

The actual cause is the **prefill top-k `argsort` scratch buffer** on the device
with the least headroom. Note what differs from the benchmarks that reached
129,960 and 260,000 tokens without trouble:

| | long-context benchmarks on record | this server run |
|---|---|---|
| harness | `llama-batched-bench` | `llama-server` |
| tensor split | `-ts 1,1,1,1` | `-ts 1,1,1,0.85` |
| CUDA2 model buffer | ~20.7 GiB | **22.65 GiB** |
| CUDA2 headroom before prefill | ~2.6 GiB | **~0.7 GiB** |

`-ts 1,1,1,0.85` exists to free room on GPU3 for the MTP weights, and it does so
by pushing roughly 2 GiB of extra model onto GPUs 0-2. That is the same ~2 GiB
the prefill `argsort` needs at depth. The prefill top-k still uses
`ggml_argsort_top_k` (`DSV4_DECODE_RADIX_TOPK` only replaces the **decode**
selection), and its CUB scratch grows with the compressed-cache width, so the
requirement rises with context.

### 3.4 Verification: the equal split fixes it

Same binary, same 104,868-token prompt, same flags, checkpoints left at their
default — only `-ts 1,1,1,1` instead of `1,1,1,0.85`:

| split | outcome |
|---|---|
| `1,1,1,0.85`, checkpoints default | OOM at 90,112 tokens |
| `1,1,1,0.85`, checkpoints disabled | OOM at 90,112 tokens |
| `1,1,1,0.85`, checkpoints disabled, MTP on | OOM at 90,112 tokens |
| **`1,1,1,1`** | **completed** — prefill 410.97 t/s, decode **33.902 t/s** cold / **34.824 t/s** warm at ~105K |

Three independent runs failed at the *identical* token count and the fourth,
differing only in tensor split, went to completion. The attribution is
unambiguous.

Two things follow.

**1. `-ts 1,1,1,0.85` must not be used for long context on the 284B.** The
production launch command in `DS4HANDOFF.md` section 4 pairs it with
`--ctx-size 131072`, which cannot actually be filled.

**2. MTP and long context are mutually exclusive on this checkpoint.** MTP is
the only reason the split is skewed, so there is no configuration in 96 GB that
has both. This is why hypothesis A1 has been open since 2026-07-10: it was never
a missing measurement, it was an unsatisfiable configuration.

Decode at ~105K with the equal split and `DSV4_MMVQ_SMALLK=1` is **34.824 t/s**
warm. The previously recorded long-context reference was 31.2-31.6 t/s at 97K
(`DS4HANDOFF.md` section 6). That is not a controlled A/B — different harness,
context and date — so it should be read as a consistency check, not a measured
delta.

### 3.5 MTP at short context — works, and it is large

Since long context is unavailable to MTP, it was measured where it *is* viable.
946-token prompt, `-ts 1,1,1,0.85`, `DSV4_MMVQ_SMALLK=1` in both arms, greedy:

| arm | decode, 64 tokens | decode, 128 tokens warm |
|---|---:|---:|
| no MTP | 39.015 | 38.119 |
| **MTP** (`--spec-type dsv4-mtp`) | **49.459** | **50.102** |
| delta | **+26.8%** | **+31.4%** |

So MTP is worth about **+31%** decode when it can be used, on top of the ~5%
from `DSV4_MMVQ_SMALLK`. Combined with the short-context no-MTP baseline this
session started from (35.46 t/s), the short-context path is now
**35.46 -> 50.10 t/s, +41.3%**.

That makes the split decision a genuine trade, not a free choice:

| workload | best config | decode |
|---|---|---:|
| short context (agent with a small working set) | `-ts 1,1,1,0.85` + MTP + SMALLK | **50.1 t/s** |
| long context (agent over a large codebase) | `-ts 1,1,1,1`, no MTP, + SMALLK | **34.8 t/s** at 105K |

There is no configuration that gets both, and picking wrong does not degrade
gracefully — the long-context case *crashes* under the MTP split rather than
running slowly.

## 4. Recommended configuration for the agentic profile

MTP is worth +31% decode but forces a split that cannot survive long prefill, so
there is no single "balanced" answer — there are two configurations and the
choice is set by whether the session's context exceeds ~90K tokens.

For a coding agent over a large codebase, long context is the defining
constraint and the split resolves in its favour: **equal split, no MTP**. Use
the short-context configuration at the end of this section only if the working
set is known to stay small.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GGML_CUDA_P2P=1 \
DSV4_CONSTANT_SHAPE=1 DSV4_SPARSE_FA=1 DSV4_FA_UNION=1 DSV4_IDX_SKIP=1 \
DSV4_MOE_TILE=1 DSV4_MOE_RESIDENT=1 DSV4_GLU_FUSE=1 DSV4_MOE_FUSE=1 \
DSV4_DECODE_FUSED_IDX=1 DSV4_DECODE_RADIX_TOPK=1 \
DSV4_MMVQ_SMALLK=1 \
DSV4_EXPERT_PARALLEL=1 DSV4_EP_TO_LAYER=1 \
build-v4-cuda/bin/llama-server \
  -m /mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -ngl 999 --split-mode layer --flash-attn on --no-repack \
  --ctx-size 131072 --batch-size 4096 --ubatch-size 512 \
  -t 8 --poll 100 -ts 1,1,1,1 --parallel 1 \
  --host 0.0.0.0 --port 18080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget 2048
```

Changes from the command in `DS4HANDOFF.md` section 4:

- `-ts 1,1,1,1` instead of `1,1,1,0.85` — **required**, see 3.3/3.4;
- drop `DSV4_MTP_SPEC` / `--spec-type dsv4-mtp` — incompatible with the above;
- add `DSV4_MMVQ_SMALLK=1` — +5% decode, prefill flat;
- `DSV4_EXPERT_PARALLEL=1 DSV4_EP_TO_LAYER=1` for the one-shot fast ingest
  (500 t/s prefill, then a 4.5 s device relayout into layer decode). Untested in
  combination with `DSV4_MMVQ_SMALLK`; the two touch different code paths (EP
  affects placement and the MoE scheduler, `SMALLK` only the MMVQ decode
  geometry), but that has not been measured together.

**Short-context variant** (working set stays well under 90K): keep
`-ts 1,1,1,0.85`, add `DSV4_MTP_SPEC=1`, `DSV4_MTP_GGUF=...` and
`--spec-type dsv4-mtp`, and lower `--ctx-size` to something the split can
actually fill. This is the 50.1 t/s configuration.

**Alternative worth considering: the K144 REAP 162B checkpoint.** It reached
500.16 t/s prefill and 31.24 t/s decode at 130K with a peak of only ~14.8 GiB
per GPU (`DS4_REAP_K144_2026-07-21.md`) — roughly 9 GiB more headroom per card
than the 284B. That headroom is exactly what the 284B ran out of here, so the
long-context ceiling does not arise. It has no MTP weights, so nothing is given
up by dropping MTP.

## 4a. Highest-value follow-up: radix-select for the prefill top-k

Everything in section 3.3/3.4 traces back to one allocation: the CUB scratch
inside `ggml_argsort_top_k` on the prefill path. `DSV4_DECODE_RADIX_TOPK`
already replaced the equivalent decode selection with an exact cooperative
radix-select that is **3.7x faster (35.4 us vs 131 us at N=32768)** and needs a
fixed, small workspace instead of a context-scaled one.

Extending that kernel to the prefill shape (batched over the ubatch's query
rows rather than a single row) would plausibly:

1. **remove the 90K ceiling outright**, since the context-scaled allocation is
   what fails;
2. **restore MTP + long context as a viable pair**, which is currently the
   single largest decode win left on the table (+31%);
3. possibly speed prefill up, though that is secondary — prefill is
   compute-bound (see H2), so the argsort is unlikely to be a large share.

Items 1 and 2 are the point. This is the recommended next piece of work and it
reuses a kernel that already exists and is already validated for exactness.

## 5. Reproduce

Kernel A/B (no rebuild needed between arms — `DSV4_MMVQ_SMALLK` is read at
runtime):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 GGML_CUDA_P2P=1
export DSV4_CONSTANT_SHAPE=1 DSV4_SPARSE_FA=1 DSV4_FA_UNION=1 DSV4_IDX_SKIP=1
export DSV4_MOE_TILE=1 DSV4_MOE_RESIDENT=1 DSV4_GLU_FUSE=1 DSV4_MOE_FUSE=1
export DSV4_DECODE_FUSED_IDX=1 DSV4_DECODE_RADIX_TOPK=1
build-v4-cuda/bin/llama-batched-bench \
  -m /mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -ngl 999 --split-mode layer --flash-attn on --no-repack \
  --ctx-size 16384 --batch-size 4096 --ubatch-size 512 \
  -t 8 --poll 100 --tensor-split 1,1,1,1 --parallel 1 \
  -npp 512,8192 -ntg 256 -npl 1 --output-format jsonl
```

Re-run the same command with `DSV4_MMVQ_SMALLK=1` for the H5 arm. Always load
the model from `/mnt/nvme`: the `/mnt/ssd` copy of the same 86.7 GB file is on
SATA at about 174 MB/s and takes **15+ minutes** to load versus about 45 s,
which dominates A/B iteration cost.

Static register/spill check without running anything:

```bash
nvcc $(python3 -c "import json;print([e['command'] for e in json.load(open('build-v4-cuda/compile_commands.json')) if e['file'].endswith('mmvq.cu')][0])" | sed 's/-o .*//') -Xptxas -v -o /dev/null 2>&1 | grep -c "bytes stack frame"
```

## 6. Not attempted, and why

- **Full upstream merge.** `#24127` rewrote `mmq.cuh` (4214 lines) into
  `mmq-load-tiles.cuh` / `mmq-vec-dot.cuh` / per-architecture config headers.
  It is performance-neutral on Ampere but would collide directly with
  `DSV4_MOE_TILE` and `DSV4_MOE_RESIDENT`, which live in exactly those files.
  Targeted cherry-picks are the right granularity.
- **PDL, NVFP4 W4A4, Blackwell MMA, Volta/Turing CUDA graphs, RDNA/CDNA
  tuning.** All recent upstream CUDA work that does not apply to sm_86.
