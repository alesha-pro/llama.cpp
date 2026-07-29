# DeepSeek-V4-Flash 284B on 4x RTX 3090 — 2026-07-27 changes

Two new kernel-level options and one launch-configuration correction. Together
they raise long-context decode from ~31 t/s to **41-43 t/s**, and lift the
usable context from a hard crash at 90K tokens to a verified **163,840**.

Target profile: a single interactive agentic user — one large context ingest,
then many rounds of decode plus small tool-result prefills. In such a session
decode is roughly 64% of wall clock, so decode is the optimization target under
the constraint that prefill must not regress.

---

## 1. `DSV4_MMVQ_SMALLK=1` — +5% decode

### What it does

Relaxes the `small_k` trigger in the CUDA MMVQ dispatcher from `<` to `<=`.

### Why it works

`mul_mat_vec_q` picks between two layouts. The default gives each block one
output row; the `small_k` variant gives it `nwarps` rows. Upstream triggers
`small_k` when "the full thread block covers all K blocks in a single loop
iteration", implemented as:

```c
use = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
```

Both DSV4 routed expert projections land **exactly on that boundary**, and the
strict `<` excludes them:

| tensor | quant | ncols_x | blocks/row | threshold | `<` | `<=` |
|---|---|---:|---:|---:|---|---|
| routed up/gate | IQ2_XXS | 4096 | 16 | 16 | false | **true** |
| routed down | Q2_K | 2048 | 8 | 8 | false | **true** |
| attention / shared / output | Q8_0 | 4096 | 128 | 32 | false | false |

Not a coincidence: `n_embd = 4096` and `n_ff_exp = 2048` with these two 2-bit
quants both put the reduction dimension at exactly what a 4-warp block covers in
one iteration. At equality the default layout does one iteration of MAC work and
then pays a full cross-warp shared-memory reduction plus `__syncthreads` for it;
`small_k` amortizes that reduction over 4x more work.

This is a **decode-only** change by construction — prefill goes through MMQ, not
MMVQ.

### Measured

Production 284B, 4x3090, three runs per arm, interleaved in time:

| shape | before | after | delta |
|---|---:|---:|---:|
| pp512 + tg256 | 36.009 | **37.971** | **+5.45%** |
| pp8192 + tg256 | 33.886 | **35.549** | **+4.91%** |

Ranges do not overlap: the three repeats span 0.13 and 0.09 t/s while the gap to
the baseline arm is 1.96 and 1.66 t/s. Prefill unchanged (582.29 vs 578.36 and
555.80 vs 556.57 — within its noise band).

`test-backend-ops -o MUL_MAT_ID`: 764/764 with the flag set.

### Also ported: upstream 683f0c72e

NVIDIA's MMVQ fix — index the accumulators by the compile-time row instead of
`threadIdx.x` so they stay in registers, and add the zero-initialized biases
unconditionally. Verified with `nvcc -Xptxas -v`: every local-memory spill in the
translation unit disappears (max stack frame 64 B -> 0 B, 159 of 264 kernels
changed).

**It is neutral for DSV4 on its own** — working out the dispatch shows our shapes
run the `small_k = false` variant, which was not spilling. It is kept because it
is a faithful upstream port, statically better or equal everywhere, and it
removes the 32-byte spill that the `small_k` variant *does* carry, which is
exactly the variant `DSV4_MMVQ_SMALLK` switches onto. Do not report it as a
speedup.

---

## 2. `DSV4_PREFILL_RADIX_TOPK=1` — removes the 90K prefill ceiling

### The bug

The prefill indexer top-k used `ggml_argsort_top_k`, which is not an op:

```c
struct ggml_tensor * ggml_argsort_top_k(ctx, a, k) {
    result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);   // sorts ALL of it
    result = ggml_view_4d(ctx, result, k, ...);            // then takes k
    return result;
}
```

So selecting 512 of 32,768 rows meant **fully sorting all 32,768** per query
token, and materializing a `[n_comp, n_tokens]` i32 result (67 MiB at
32768x512) plus CUB segmented-sort temp storage. Both scale with context.

On `-ts 1,1,1,0.85` this failed at **90,112 prompt tokens**:

```
CUDA error: out of memory   current device: 2, in function alloc (ggml-cuda.cu:533)
cuMemCreate <- argsort_f32_i32_cuda_cub <- ggml_cuda_op_argsort
```

Reproduced three times at the identical token count. It was invisible until now
because every long-context number on record was measured with
`llama-batched-bench` at the equal `1,1,1,1` split, which has ~2.6 GiB of CUDA2
headroom instead of ~0.7 GiB.

### The fix

`top_k_radix_rows` in `ggml/src/ggml-cuda/top-k.cu`: **one CUDA block per row**,
all four 8-bit radix passes kept in shared memory, then a compaction that writes
indices above the cutoff and fills the remainder from the ties. No grid
synchronization, no global histogram, **no context-scaled scratch** — the only
allocation is the `[k, nrows]` output that `GGML_OP_TOP_K` declares.

The fork already had an exact radix-select for decode (`DSV4_DECODE_RADIX_TOPK`,
35.4 us vs 131 us for a full argsort at N=32768), but it is built around a
grid-wide cooperative launch serving exactly one row. Prefill is the opposite
shape — many independent rows — hence the separate kernel. The single-row path
is untouched; one block would be far worse than the whole grid for a lone 32K
row.

`DSV4_PREFILL_RADIX_TOPK=1` routes both multi-row `deepseek4.cpp` call sites
through `ggml_top_k`.

**Order caveat:** the returned set is exact, its order is unspecified, and
`GGML_OP_TOP_K` does not promise one. Both consumers are order-invariant — the
mask builder pairs each value with its own index via `set_rows`, and the KV
gather feeds attention. Because compaction uses `atomicAdd`, order also varies
run to run, so text is not bit-reproducible across runs. This is already true of
`DSV4_DECODE_RADIX_TOPK`.

### Correctness

`test-backend-ops -o TOP_K`: **455/455** against the CPU reference. The existing
suite already swept multi-row shapes up to `ncols = 524,299` at `nrows = 2` with
`k` up to 9999, so the batched path was covered from the start; what it lacked
was the production combination of `k = 512` with a realistic row count. Eight
such cases were added:

```
TOP_K(ne=[8192|32768|32771, 8|512, 1, 1], k=512, ties=0|1)   all OK
```

`ne0 = 32771` is deliberately not a multiple of the 256-thread block size, and
the `ties` variants matter because every masked position carries the same
`-INF`/`-1e30` sentinel, so ties at the cutoff are the common case here.

### Speed

Prefill is **unchanged** (+0.5% at 32K, noise). Expected: DSV4 prefill is
compute-bound on MMQ — one ubatch touches essentially the whole expert weight
set, about 107 ms of pure DRAM time against ~1,034 ms actual — so the selection
was never a large share. The change is about the allocation, not the clock.

---

## 3. Use `-ts 1,1,0.90,0.95`, not `1,1,1,0.85`

Fixing prefill exposed a second ceiling. At `-ts 1,1,1,0.85` with MTP and
`--ctx-size 131072`:

| GPU | free | model | context | compute | unaccounted |
|---|---:|---:|---:|---:|---:|
| CUDA0 | 480 | 22617 | 219 | 409 | 397 |
| CUDA1 | 1894 | 20779 | 258 | 787 | 403 |
| CUDA2 | **10** | 22652 | 261 | 793 | 405 |
| CUDA3 | 2452 | 15638 | 174 | 822 | 5035 (MTP) |

A 114,086-token prompt **completes prefill and then dies in decode**:

```
dsv4-mtp: state shadows ready (43 layers)
CUDA error: out of memory
  in function ggml_cuda_graph_evaluate_and_capture ... cudaGraphLaunch
```

The MTP decode CUDA graph cannot be instantiated in CUDA2's 10 MiB.

`-ts 1,1,1,0.85` exists only to free GPU3 for the MTP weights. With MTP now
viable at depth, that skew starves the device that has to host the deepest decode
graph. Layer granularity is ~1,900 MiB, so the useful move is to shift exactly
one layer off CUDA2:

| GPU | model before | model after | free before | free after |
|---|---:|---:|---:|---:|
| CUDA0 | 22617 | 22617 | 480 | **684** |
| CUDA1 | 20779 | 20779 | 1894 | **2260** |
| CUDA2 | 22652 | **20750** | **10** | **2312** |
| CUDA3 | 15638 | **17540** | 2452 | **774** |

Total model footprint is identical (81,686 MiB either way) — pure
redistribution. Minimum free VRAM across devices: **10 MiB -> 684 MiB**.

---

## 4. Results

All: full 284B IQ2_XXS checkpoint from NVMe, 4x RTX 3090 at 220 W, layer split,
`ubatch 512`, FlashAttention, P2P, `--parallel 1`, MTP enabled unless stated,
`llama-server`.

### Depth vs speed

| prompt depth | prefill | decode (MTP) | decode (no MTP) |
|---:|---:|---:|---:|
| ~1K | — | **50.1 t/s** | 38.1 t/s |
| 105K | 414 t/s | **41.6 t/s** | 34.6 t/s |
| 127K (ctx 131072 full) | 396 t/s | **42.6 t/s** | — |
| 156K (ctx 163840 full) | 367 t/s | **41.4 t/s** | — |

MTP is worth **+31%** at short context and **+20%** at depth — smaller deep
because a larger share of each token goes to the indexer and attention, which
speculation does not amortize.

### Maximum context

| ctx | outcome |
|---:|---|
| 131072 | fills — 127,356 tokens, 395.93 t/s prefill, 42.648 t/s decode |
| **163840** | **fills — 156,423 tokens, 367.19 t/s prefill, 41.386 t/s decode** |
| 262144 | **fails** at 204,800 tokens: 1,189 MiB compute buffer on CUDA3 |

CUDA3 is the binding device because it carries the MTP weights:
`24123 - 17540 model - 4957 MTP = 1626 MiB` for KV + compute. KV there is
0.00164 MiB/token. The compute buffer scales with depth; from the single deep
measurement (1,189.40 MiB at 204,800) the fill limit works out to **~205-220K**,
less allocator overhead.

The model reproduced the failure rather than being fitted to it: at ctx 262144
the budget was `1626 - 430 = 1196` MiB against a 1,189.40 MiB request — a
**+7 MiB** margin, and it failed. It then correctly predicted that 163840 (margin
279-405 MiB) would work. 174080 is projected to fit; 196608 is the untested
boundary case at ~140-160 MiB.

**Untested lever:** `--ubatch-size 256` roughly halves the compute buffers and by
the same model would allow ~357K, at some prefill cost.

### Where this sits historically

| configuration | context | decode |
|---|---:|---:|
| previously recorded reference, no MTP | ~97K | 31.2-31.6 |
| equal split, no MTP, `+SMALLK` | ~105K | 34.8 |
| `+PREFILL_RADIX_TOPK` + MTP, `1,1,1,0.85` | ~105K | 41.6 |
| **`1,1,0.90,0.95`, all of the above** | **156K** | **41.4** |

---

## 5. Production launch

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GGML_CUDA_P2P=1 \
DSV4_CONSTANT_SHAPE=1 DSV4_SPARSE_FA=1 DSV4_FA_UNION=1 DSV4_IDX_SKIP=1 \
DSV4_MOE_TILE=1 DSV4_MOE_RESIDENT=1 DSV4_GLU_FUSE=1 DSV4_MOE_FUSE=1 \
DSV4_DECODE_FUSED_IDX=1 DSV4_DECODE_RADIX_TOPK=1 \
DSV4_MMVQ_SMALLK=1 DSV4_PREFILL_RADIX_TOPK=1 \
DSV4_MTP_SPEC=1 \
DSV4_MTP_GGUF=/mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
build-v4-cuda/bin/llama-server \
  -m /mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -ngl 999 --split-mode layer --flash-attn on --no-repack \
  --ctx-size 163840 --batch-size 4096 --ubatch-size 512 \
  -t 8 --poll 100 -ts 1,1,0.90,0.95 --parallel 1 --spec-type dsv4-mtp \
  --host 0.0.0.0 --port 18080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget 2048
```

Three changes from what `DS4HANDOFF.md` section 4 recorded: `-ts 1,1,0.90,0.95`
instead of `1,1,1,0.85`, plus the two new flags. Without
`DSV4_PREFILL_RADIX_TOPK=1` this command dies at 90,112 tokens.

Always load the model from `/mnt/nvme`. The `/mnt/ssd` copy of the same 86.7 GB
file is on SATA at ~174 MB/s and takes 15+ minutes to load versus ~45 s.

---

## 6. Rejected, with the arithmetic

Recorded because the reasoning is reusable.

**Dedup of MoE activation quantization** (upstream #25441). For MoE gate/up the
activation is broadcast across experts, so each token is re-quantized once per
routed slot; upstream adds a quantize-once-and-scatter path. Computed before
implementing: per MoE layer per ubatch of 512 the read drops from 64 MiB to
8 MiB, about 2.24 GiB per ubatch across ~40 routed layers, ~560 MiB per GPU,
about **0.75 ms** at 750 GB/s — against a ubatch that takes **1,034 ms**. That
is 0.07%, far below noise. The deeper reason: prefill here is compute-bound, not
bandwidth-bound. Not implemented.

**Vectorized `get_rows`** (upstream #25929/#25962). The decode gather moves about
25 MiB per token over 21 CSA layers, roughly **0.03 ms** against a ~32 ms token.
Also a F16-to-F32 cast, which upstream's vectorized path explicitly excludes.
Not implemented.

**Server context checkpoints.** They grow at ~6.7 KiB/token (482 -> 536 -> 590
MiB observed) and looked like the obvious cause of the 90K OOM.
`--checkpoint-every-n-tokens -1` failed at the **identical** token count — they
live in host RAM (`--cache-ram`), not VRAM. Hypothesis refuted.

**Full upstream merge.** Upstream `#24127` rewrote `mmq.cuh` (4,214 lines) into
per-architecture headers. Performance-neutral on Ampere, but it collides
directly with `DSV4_MOE_TILE` and `DSV4_MOE_RESIDENT`, which live in exactly
those files. Targeted cherry-picks are the right granularity.

Upstream also **removed `-sm row`** entirely (#24216), independently confirming
the fork's earlier finding that row split is not viable for DSV4, and
**reverted** its own attempt to reduce scheduler synchronizations (#20793) — so
no ready-made fix exists upstream for the host-sync overhead documented in
`docs/ds4/DS4_DECODE_UPLIFT_RESEARCH_2026-07-10.md`.

---

## 7. Where the fork still differs from upstream

Upstream (`0e4a03622`, 2026-07-27) has since implemented several DSV4 pieces the
fork already had, but on the long-context path the fork is **not** behind:

| area | upstream | this fork |
|---|---|---|
| indexer top-k applied to attention | `-INF` fill + `set_rows` zeros, then **dense FA over the whole KV cache** (O(n_kv)) | physical gather of only the selected 512 rows (O(top_k)) |
| Top-512 selection | `ggml_top_k` (CUB / bitonic argsort) | exact cooperative radix-select, 35.4 us vs 131 us at N=32768 |
| Lightning Indexer op | 588 lines, vec + wmma | 930 lines: adds q-tiled, fp16-acc, causal-skip |
| MoE tiling | worst-case `ncols_max` | real per-expert bound + device-resident scheduler |
| expert parallel / EP-to-layer | none | opt-in, one-shot device relayout |

Upstream's dense-FA-with-mask means its attention cost still grows with full
context length; the fork's gather keeps it at the top-k width. That is the main
reason these long-context numbers are not reachable upstream.

---

## 8. Analysed but not run

### `MMVQ_PARAMETERS_TURING` — deprioritized, it would undo part of section 1

Upstream `d7be46189` added a Turing+ nwarps table the fork lacks; Ampere would
use it. It changes only K-quant shapes at `ncols_dst = 1`, dropping nwarps from
4 to 2 — which for us means `ffn_down` (Q2_K). That looked like a free port until
the interaction with `DSV4_MMVQ_SMALLK` was worked out:

| table | nwarps | threshold | blocks/row | `small_k` under `<=` |
|---|---:|---:|---:|---|
| GENERIC (current) | 4 | 8 | 8 | **true** |
| TURING (upstream) | 2 | 4 | 8 | **false** |

`small_k`'s threshold is `nwarps * blocks_per_iter_1warp`, so halving nwarps
halves the threshold and pushes `ffn_down` back out of the variant that section 1
measured as a win. The two changes are not independent: adopting the table would
**undo half of the +5%**, and its own merit at nwarps=2 is unmeasured.

That turns a one-line port into a 2x2 experiment (table x `SMALLK`) needing two
builds, since the device table id is compile-time and baked into
`__launch_bounds__`. Not worth it against an unknown while a measured win is on
the table. Revisit only if the MMVQ decode path is revisited wholesale.

### Agentic round-trip — the measurement gap

Everything measured so far is *one* large prefill plus a decode burst. The actual
agentic loop is different and has never been profiled here: after the initial
ingest, each turn appends a tool result (a few hundred to a few thousand tokens)
and decodes a response, tens of times, against a monotonically growing cache.

Open questions, all cheap to answer once a GPU slot is free:

1. Does `cache_prompt` reuse the whole prefix at depth, or is any of it
   re-prefilled? A re-prefill at 150K depth would dominate everything else in
   this document.
2. What is the per-turn latency breakdown (incremental prefill vs decode) at
   ~30K, ~100K and ~150K depth?
3. Do the small incremental prefills (200-3000 tokens) still get the
   `DSV4_MOE_RESIDENT` / `DSV4_MOE_TILE` paths, which gate on `ne12 >= 64`?

Harness written and ready: `scripts/ds4-agentic-roundtrip.py`.

## 9. Open items

- **196608 context** — projected margin ~140-160 MiB, untested. The only
  remaining question about the ceiling.
- **`--ubatch-size 256`** — would roughly halve compute buffers and by the model
  allow ~357K; costs prefill speed by an unmeasured amount.
- **MTP acceptance rate at depth** — not instrumented; the +20% is end-to-end
  throughput, not a measured accept rate.
- **Other context-scaled buffers** — the `[1, n_comp, n_tokens]` mask in the
  dense branch was not audited; it is not on the `DSV4_SPARSE_FA` path used in
  production.
- **`MMVQ_PARAMETERS_TURING`** — upstream added a Turing+ nwarps table (#23729)
  the fork lacks; it changes only K-quant `ncols_dst=1` shapes and would need a
  separate build to A/B.
