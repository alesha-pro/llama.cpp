# DS4 prefill: exact batched radix-select Top-K (`DSV4_PREFILL_RADIX_TOPK`)

Date: 2026-07-27. Follow-up to `DS4_UPSTREAM_KERNEL_PORT_2026-07-27.md`
section 4a, which identified this as the highest-value remaining work.

## The problem

`DS4_UPSTREAM_KERNEL_PORT_2026-07-27.md` section 3.3/3.4 established that
`llama-server` on the full 284B checkpoint with the production
`-ts 1,1,1,0.85` split dies at **90,112 prompt tokens**:

```
CUDA error: out of memory   current device: 2, in function alloc (ggml-cuda.cu:533)
cuMemCreate(&handle, reserve_size, &prop, 0)
  <- argsort_f32_i32_cuda_cub <- ggml_cuda_op_argsort
```

Reproduced three times at the identical token count; the equal `1,1,1,1` split
completed the same prompt. Because `-ts 1,1,1,0.85` exists only to make room for
the MTP weights, that made **MTP and long context mutually exclusive** — and MTP
is worth +31.4% decode.

Root cause: the prefill indexer top-k used `ggml_argsort_top_k`, which is not an
op — it is `ggml_argsort` over the full compressed width followed by a view of
the first `k`:

```c
struct ggml_tensor * ggml_argsort_top_k(ctx, a, k) {
    struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);
    result = ggml_view_4d(ctx, result, k, ...);
    return result;
}
```

So selecting 512 of 32,768 rows meant **fully sorting all 32,768**, per query
token, and materializing:

- a `[n_comp, n_tokens]` i32 result — 32768 x 512 x 4 B = **67 MiB**;
- CUB's segmented-radix-sort temp storage for 16.7M key/value pairs.

Both scale with context, which is exactly the wrong shape for a fixed VRAM
budget.

## The change

Two parts, both opt-in behind `DSV4_PREFILL_RADIX_TOPK=1`.

**1. A batched exact Top-K CUDA kernel** (`top_k_radix_rows` in
`ggml/src/ggml-cuda/top-k.cu`). The fork already had an exact radix-select for
the single-row decode case (`DSV4_DECODE_RADIX_TOPK`, 35.4 us vs 131 us for a
full argsort at N=32768), but it is built around a grid-wide cooperative launch
that serves exactly one row. The prefill shape is the opposite: many independent
rows, each modest.

So the batched kernel uses **one CUDA block per row**, keeping all four 8-bit
radix passes in shared memory:

- `__shared__ unsigned int hist[256]` plus five scalars for the running prefix,
  prefix mask, count-above, and the two compaction counters;
- four passes narrowing the key prefix 8 bits at a time, each pass counting only
  the keys that still match the prefix;
- a final compaction writing indices `> threshold` first and then filling the
  remainder from the `== threshold` ties.

No grid synchronization, no global histogram, **no context-scaled scratch at
all** — the only allocation is the `[k, n_tokens]` output that `GGML_OP_TOP_K`
declares. Dispatch requires `nrows > 1 && ncols >= k && ncols >= 512`; the
single-row cooperative path is untouched, since one block would be far worse
than the whole grid for a lone 32K row.

**2. Route the prefill graph through `GGML_OP_TOP_K`.** Both multi-row call
sites in `src/models/deepseek4.cpp` — the dense compressed-mask path and the
sparse-FA prompt-chunk path — switch from `ggml_argsort_top_k` to `ggml_top_k`
when the flag is set.

Order caveat, same as the decode path: the returned set is exact, its order is
unspecified, and `GGML_OP_TOP_K` does not promise one. Both consumers are
order-invariant — the mask builder scatters `values[j]` to `mask[topk[j]]` with
`ggml_set_rows`, pairing each value with its own index, and the KV gather feeds
attention, which is order-invariant over gathered keys. Because compaction uses
`atomicAdd`, the *order* also varies run to run, so generated text is not
bit-reproducible across runs; this is already true of `DSV4_DECODE_RADIX_TOPK`.

## Correctness

`test-backend-ops -b CUDA0 -o TOP_K`: **455/455 pass**, validated against the
CPU reference.

The existing suite already exercised this kernel more than expected — the
`{(1<<i)+11, 1, 2, 1}` family sweeps `ncols` up to 524,299 at `nrows = 2` with
`k` up to 9999, including tie variants, so the batched path was covered for
correctness from the start. What it did **not** contain was the production
combination: `k = 512` together with a realistic row count. Eight such cases
were added to `tests/test-backend-ops.cpp`:

```
TOP_K(ne=[8192,8,1,1],   k=512, ties=0)   OK
TOP_K(ne=[32768,8,1,1],  k=512, ties=0)   OK
TOP_K(ne=[32768,8,1,1],  k=512, ties=1)   OK
TOP_K(ne=[32771,8,1,1],  k=512, ties=1)   OK
TOP_K(ne=[8192,512,1,1], k=512, ties=0)   OK
TOP_K(ne=[32768,512,1,1],k=512, ties=0)   OK
TOP_K(ne=[32768,512,1,1],k=512, ties=1)   OK
TOP_K(ne=[32771,512,1,1],k=512, ties=1)   OK
```

`ne0 = 32771` is deliberately not a multiple of the 256-thread block size, and
the `ties=1` variants matter because every masked position in the real indexer
scores carries the same `-INF`/`-1e30` sentinel, so ties at the cutoff are the
common case here, not an edge case. `n_tokens = 512` also checks the grid
dimension actually used in production, where the earlier coverage only ever ran
two rows.

## Results

### Prefill throughput: unchanged

32,768-token prompt, equal split, two runs per arm:

| arm | prefill | decode |
|---|---:|---:|
| argsort (off) | 484.93 / 483.71 | 35.468 / 35.705 |
| radix (on) | 487.65 / 485.55 | 34.970 / 35.559 |

+0.5% prefill, within noise. This was expected and is not the point: DSV4
prefill is compute-bound on MMQ (see H2 in the companion document), so the
selection was never a large share of the time. The change is about the
allocation, not the clock.

### The 90K ceiling: gone

The decisive test is the configuration that failed three times — production
`-ts 1,1,1,0.85`, `--ctx-size 131072`, the same 104,868-token prompt through
`llama-server`:

| split | prefill top-k | outcome |
|---|---|---|
| `1,1,1,0.85` | argsort | OOM at 90,112 tokens |
| `1,1,1,0.85` | argsort, checkpoints off | OOM at 90,112 tokens |
| `1,1,1,0.85` | argsort, checkpoints off, MTP on | OOM at 90,112 tokens |
| **`1,1,1,0.85`** | **radix** | **completed** — prefill 416.48 t/s, decode 34.390 cold / 34.587 warm |

Verified from the server log that the path was actually taken
(`dsv4: DSV4_PREFILL_RADIX_TOPK=1 - exact radix-select Top-K for prefill`),
that prefill reached `progress = 0.999962`, and that the OOM count is zero.

Decode at ~105K is **34.587 t/s** on the skewed split, essentially matching the
34.824 t/s the equal split reached — so nothing was given up to get here.

### MTP + long context: possible for the first time

With the ceiling gone, the split no longer has to be chosen against MTP. Same
`-ts 1,1,1,0.85`, same 104,868-token prompt, MTP enabled:

| arm (all `-ts 1,1,1,0.85`, ~105K prompt) | prefill | decode 64 | decode 128 warm |
|---|---:|---:|---:|
| no MTP | 416.48 | 34.390 | 34.587 |
| **MTP** | 414.20 | **41.035** | **41.624** |
| delta | -0.5% | +19.3% | **+20.3%** |

`DS4_UPSTREAM_KERNEL_PORT_2026-07-27.md` section 3.4 concluded that "there is no
configuration that gets both". That is no longer true — this is that
configuration.

Long-context decode over the whole line of work:

| configuration | decode at ~100K |
|---|---:|
| previously recorded reference, no MTP (`DS4HANDOFF.md` section 6) | 31.2-31.6 |
| equal split + `DSV4_MMVQ_SMALLK` | 34.824 |
| **`-ts 1,1,1,0.85` + `SMALLK` + `PREFILL_RADIX_TOPK` + MTP** | **41.624** |

MTP's long-context gain (+20.3%) is smaller than its short-context gain
(+31.4%, measured at 946 tokens), which is expected: at depth a larger share of
each token goes to the indexer and attention, which speculation does not
amortize.

## Recommended configuration, updated

This supersedes section 4 of `DS4_UPSTREAM_KERNEL_PORT_2026-07-27.md`.

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
  --ctx-size 131072 --batch-size 4096 --ubatch-size 512 \
  -t 8 --poll 100 -ts 1,1,1,0.85 --parallel 1 --spec-type dsv4-mtp \
  --host 0.0.0.0 --port 18080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget 2048
```

Both new flags are required for this to work: `DSV4_PREFILL_RADIX_TOPK=1` is
what makes `-ts 1,1,1,0.85` survive a long prefill, and without it this exact
command OOMs at 90,112 tokens.

## Caveats and open items

- Validated to ~105K. **A follow-up run at 114,086 tokens got through prefill
  and then failed in decode** — `cudaGraphLaunch` OOM on CUDA2 at
  `ggml_cuda_graph_evaluate_and_capture`, i.e. the MTP decode graph could not be
  instantiated in the 10 MiB CUDA2 had left. So this change lifts the *prefill*
  ceiling, but a second, lower ceiling sits in decode-graph capture. See
  `DS4_MAX_CONTEXT_MTP_2026-07-27.md`.
- MTP acceptance rate at depth was not instrumented — only end-to-end
  throughput. The +20.3% is the observable, not a measured accept rate.
- Output ordering is non-deterministic (atomic compaction), so runs are not
  bit-reproducible. Set-exactness is what the tests verify.
- The batched kernel re-reads the row from global memory on each of the four
  passes plus the compaction. At 32K x 512 that is five streaming passes; a
  candidate-compaction optimization after pass 1 was not attempted because
  prefill time did not move and the goal was the allocation.
