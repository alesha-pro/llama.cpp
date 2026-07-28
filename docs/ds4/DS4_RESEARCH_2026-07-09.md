# DeepSeek-V4-Flash on 4×RTX 3090: upstream, Unsloth, and parallelism research

Date: 2026-07-09 UTC

Repositories examined:

- performance fork: `/mnt/ssd/engines/llama.cpp-v4-cchuter`, branch `ds4-longctx`, HEAD `06e8035`;
- current upstream checkout: `/mnt/nvme/engines/llama.cpp`, upstream HEAD `049326a00`;
- local TP research branch: `codex/ds4-tensor-parallel` in the current upstream checkout.

No commit, push, or public PR was made. The TP code is a local research prototype.

## Executive conclusion

The fork is not obsolete. Unsloth's DeepSeek-V4 work and the fork solve different problems:

- Unsloth and upstream made the architecture usable, converted/quantized the weights, supplied templates, and fixed correctness issues such as quantized compressed-cache rotation.
- The fork changes the runtime hot paths for this exact 4×3090 machine: sparse physical K/V gather, a fused lightning indexer, correct small-MoE tiling, constant-shape decode, and MTP.

On the same 80.76 GiB GGUF and the same host, the fork is already substantially faster than current upstream and supports contexts that current upstream cannot allocate. At 2K and 8K prompts the measured fork lead is 43–58% in prefill and 63–97% in decode. Current upstream fails while creating a 100K context, and later fails during a 32K test; the fork completes 98K at 433 t/s and has already completed 253K.

So the right strategy is not to discard the fork. Keep it as the production/performance branch, and forward-port its proven ideas in small, benchmarked groups onto current upstream so that future architecture fixes do not have to be reimplemented forever.

Classic four-way width tensor parallelism is now implemented in the local research branch and produces the same deterministic tokens as layer split on a real-weight four-layer fixture. It is nevertheless not the production answer on this rig. The cards have no CUDA P2P path and no NVLink; NCCL reports `SHM/direct/direct`, so each transformer layer pays host-mediated collectives. The full model made TP 40.7% slower in prefill and 69.5% slower in decode than current upstream layer split, even though all GPUs were involved concurrently.

For this machine, the highest-value next work is a combination of:

1. device-resident/persistent MoE scheduling, eliminating the host readback used to discover the live expert tile bound;
2. adaptive K=2 MTP and larger verification batches for single-stream decode;
3. a DS4-aware expert-parallel experiment for prefill, measured against width TP rather than assumed faster;
4. a microbenchmark and possible pinned-host collective specialized for 16 KiB decode reductions;
5. staged forward-porting of the fork's already-proven kernels to current upstream.

## What Unsloth actually contributed

The Unsloth DeepSeek-V4 page uses llama.cpp as the inference runtime. It is not evidence of a separate Unsloth CUDA engine that supersedes this fork. The important Unsloth-facing contributions are:

- correct model conversion and GGUF quantization, including MXFP4/FP8 details;
- usable chat/reasoning templates and launch guidance;
- a catalogue of quality/size tradeoffs;
- identification and upstreaming of the compressed-cache quantization/rotation fix in PR #25202.

Relevant upstream state:

- PR #24162 is the main DeepSeek-V4 architecture implementation. Its own TODO list included MTP, tensor split, lightning-indexer work, and dedicated hyperconnection/sinkhorn operations.
- PR #25202 fixes correctness for quantized compressed caches. It is important if the fork starts using those cache types, but it is not a general inference-speed implementation.
- PR #24231, the dedicated lightning-indexer operation, was still open during this research. Its visible patch was principally the operation/API/CPU/tests; the current fork already has a purpose-built fused CUDA path and avoids the multi-gigabyte materialization.

Unsloth's published quality table also matters. The antirez IQ2_XXS quant used here is much smaller and lower-quality than Unsloth's recommended quants: their table reports roughly 6.08 perplexity for antirez IQ2_XXS versus roughly 4.53 for the unquantized reference. Therefore two claims must not be conflated:

- on this exact small GGUF, the fork already beats the current Unsloth-recommended upstream runtime;
- it has not yet proved equal speed at the quality level of Unsloth UD-IQ3_XXS/UD-Q4_K_XL. Those files also do not fit wholly in 96 GB VRAM.

Sources:

- <https://unsloth.ai/docs/models/deepseek-v4>
- <https://github.com/ggml-org/llama.cpp/pull/24162>
- <https://github.com/ggml-org/llama.cpp/pull/25202>
- <https://github.com/ggml-org/llama.cpp/pull/24231>

## Reproduced same-machine A/B

Model:

`/mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

Common important settings: CUDA, FlashAttention on, all layers offloaded, layer split, no repack, batch 4096, ubatch 512, 8 CPU threads. The fork additionally enabled its proven `DSV4_*` paths.

| Prompt/test | Current upstream pp | Fork pp | Fork delta | Current upstream tg32 | Fork tg32 | Fork delta |
|---|---:|---:|---:|---:|---:|---:|
| 2,048 tokens | 344.47 t/s | 492.52 t/s | +43.0% | 14.87 t/s | 24.29 t/s | +63.3% |
| 8,192 tokens | 332.19 t/s | 523.82 t/s | +57.7% | 15.67 t/s | 30.90 t/s | +97.2% |
| 32,768 tokens | OOM during run | 467.87 t/s | capability win | OOM | 25.25 t/s | capability win |
| 98,304 tokens | context cannot be created | 433.47 t/s | capability win | unavailable | 16.97 t/s | capability win |

Caveats:

- Upstream could only reserve a 33,024 context for the short rows and disabled pipeline mode after a compute-buffer OOM. A 100,352 context failed immediately. The fork used a 100,352 context for the table.
- `llama-batched-bench` repeats synthetic token IDs. In the first hash-routed layers this creates unusually concentrated routing; the fork logged an actual maximum of 512 columns versus the 4096 conservative bound. Real prompts are the authoritative MoE measurement.
- The comparison is still useful because it uses the exact same GGUF, hardware, FlashAttention setting, batch/ubatch, and no-repack policy.

The current upstream graph explains the difference:

- its lightning indexer is a chain of matmul, permutation/contiguous materialization, ReLU, multiply, row sum, and top-k;
- its sparse attention is primarily mask-based rather than a physical K/V gather;
- its generic MoE MMQ sizes for the conservative maximum instead of the actual routed maximum;
- it has no DS4 MTP implementation;
- its compute buffers are several GiB at long context, where the fork's specialized graph stays in the hundreds of MiB per GPU.

## The tensor-parallel prototype

### Why generic llama.cpp TP could not simply be enabled

Before the prototype, upstream deliberately rejected `LLM_ARCH_DEEPSEEK4` in `llm_arch_supports_sm_tensor`. Bypassing that check alone fails for structural reasons:

- generic cache placement looks for `attn_output.weight`, but DS4 has `attn_output_a.weight` and `attn_output_b.weight`;
- the generic FlashAttention meta rule requires Q, K, and V all to be head-sharded;
- DS4 has 64 Q heads but one shared 512-dimensional MLA K/V head, so that K/V head cannot be divided four ways;
- the output projection is grouped into eight low-rank groups, not a conventional single `wo` matrix;
- `ROPE_BACK`, used by the DS4 de-rotation path, was not able to preserve a sharded head state;
- shared-expert tensor names did not match the generic routed-expert patterns.

### Implemented mapping

| Component | Four-GPU mapping | Reason |
|---|---|---|
| `attn_q_a`, Q-A norm | mirrored | RMS is across the 1024 Q low-rank dimension |
| `attn_q_b` | split output dimension, 16 Q heads/GPU | 64 heads divide cleanly by four |
| `attn_kv`, raw/compressed K/V caches | mirrored | there is exactly one shared MLA K/V head |
| attention sinks | 16 sinks/GPU | aligned with local Q heads |
| FlashAttention | local Q heads against mirrored K/V | correct GQA/MLA semantics |
| `attn_output_a` | two of eight output groups/GPU | preserves grouped batched matmul |
| `attn_output_b` | input-axis split, then AllReduce | reconstructs the 4096 hidden state |
| routed expert gate/up | intermediate-width split | generic width TP |
| routed expert down | input-width split, delayed AllReduce | reconstructs hidden state |
| shared expert | same up/down width split | avoids replicating 1.07 GiB of shared weights |

The meta backend was extended for:

- Q-head-sharded FlashAttention with mirrored K/V;
- batch-axis-sharded grouped matmul for `output_a`;
- sharded `ROPE_BACK`;
- DS4 tensor naming and granularity.

The result performs two hidden-state collectives per transformer layer: one after attention output-B and one after the combined expert path. In decode they are 4096 float values, or 16 KiB, but latency rather than bandwidth dominates.

### Correctness fixture

A local four-layer, 8.85 GiB GGUF was derived from the production file. It retains the exact tensors and quantization for layers 0–3, including hash routing and the first CSA/HCA compression cases, plus the original embedding/output tensors. It exists only under the ignored upstream build directory.

Validation:

- the TP graph loads and evaluates;
- deterministic greedy generation is byte-for-byte identical to layer split for the test prompt;
- both modes generated `tons … económetrics dekameters Duodecimal etxek` after the same prefix;
- the current branch passes `git diff --check` and builds `llama-batched-bench`, `llama-cli`, `llama-perplexity`, and `llama-server` with CUDA 12.8/NCCL.

Stable fixture result (`pp256`, `tg32`):

| Split | Prefill | Decode |
|---|---:|---:|
| layer | 906.57 t/s | 144.99 t/s |
| tensor | 773.67 t/s | 73.86 t/s |
| tensor relative to layer | -14.7% | -49.1% |

This is not a full-model speed prediction, but it proves that the mapping is correct and that collective cost is already larger than the saved compute on this topology.

### Full 43-layer result

After fixing a generic trailing-host-view bug in the meta backend, the full 80.76 GiB model loaded and completed a TP benchmark. Model plus the small test context used 22,214 MiB on each GPU. The actual KV depth after prompt plus generation was 544 tokens.

For a final three-way check, the same full GGUF, binary settings, ubatch 512, no-repack policy, `pp512`, `tg32`, and actual `N_KV=544` were used. Upstream layer mode needed its normal retry with pipeline parallelism disabled after the initial 5,180.78 MiB compute-buffer reservation failed. The fork retained its specialized pipeline graph.

| Engine/split | pp512 | tg32 |
|---|---:|---:|
| current upstream, layer | 295.06 t/s | 17.01 t/s |
| current upstream, new tensor prototype | 174.99 t/s | 5.18 t/s |
| optimized fork, layer, no MTP | 435.33 t/s | 34.69 t/s |

Consequences:

- TP versus upstream layer: -40.7% prefill and -69.5% decode;
- fork versus upstream layer: +47.5% prefill and +103.9% decode;
- fork versus TP: 2.49× prefill and 6.70× decode.

One-second `nvidia-smi dmon` samples during full TP usually showed all four devices awake but only roughly 0–17% SM activity, with about 118–125 W per card. This is the opposite of the hoped-for outcome: the work is distributed, but collective/launch latency creates bubbles large enough that simultaneous placement does not produce simultaneous useful arithmetic.

The TP and layer runs used different maximum context reservations because current upstream layer mode would not reserve the smaller `-b 512` graph and TP cannot afford upstream's long-context compute buffers. This does not change the actual benchmark depth (`N_KV=544`) or ubatch shape, but it is recorded here to avoid claiming a bit-for-bit identical allocator state.

### Live hardware topology

All four RTX 3090s are PCIe 3.0 ×16 devices on separate host bridges inside NUMA node 0. Live checks report:

- every GPU-to-GPU link as `NODE`;
- CUDA peer read/write status `CNS` for every pair;
- no `NV#` link in `nvidia-smi topo -m`;
- NCCL communicator `isAllCudaP2p 0`;
- NCCL channels `via SHM/direct/direct`.

The low utilization under layer split is therefore expected for a single decode stream: a token is causally dependent through 43 layers, and contiguous layer groups execute on their owning GPUs in sequence. Width TP makes the utilization graphs look fuller, but useful throughput is the metric that matters. On this host, every layer then crosses host-mediated NCCL twice.

If physical layout allows two 3090 NVLink bridges, a future 2-way TP × 2-stage pipeline experiment is more plausible than four-way TP. A 3090 cannot form a native four-card NVLink fabric; the useful design would be two independent pairs, with only stage activations crossing between pairs.

## Ranked optimization hypotheses

The estimates below are research priors, not promised gains. Each must be accepted only after same-prompt A/B and correctness checks.

### P0-A: device-resident MoE scheduling and persistent grouped MMQ

Current `DSV4_MOE_TILE` gets its large gain by using the actual routed maximum rather than the conservative 4096-column bound. The remaining architectural problem is that discovering/using that bound involves host-visible state and prevents a fully capturable, continuously scheduled MoE path.

Prototype:

1. keep expert histogram, prefix bounds, and maximum on-device;
2. launch a persistent work queue over `(expert, token tile)` jobs;
3. let the kernel read bounds instead of selecting a host-side launch shape;
4. keep an overflow/tail path for pathological routing;
5. capture the stable outer graph with CUDA graphs.

For decode, add a grouped top-6 expert GEMV specialization. A useful second stage would fuse gate/up dequantization, SwiGLU clamp, and down accumulation with a small per-expert intermediate. This targets kernel launch and under-filled-CTA overhead, not only arithmetic count.

Why first: it attacks the hottest 78 GiB of weights, removes a synchronization point, and benefits both the current layer-split production path and any future EP/TP path.

### P0-B: adaptive MTP K=2 rather than fixed K=1

K=1 already produces 1.2–1.5× decode improvement with high greedy acceptance. The branch contains parked K=2 infrastructure. Revive it as an adaptive policy:

- use K=2 only after an online acceptance window exceeds a threshold;
- fall back immediately on tool/grammar boundaries and after rejection;
- verify two candidates in one stable graph/batch;
- record accepted tokens per main-model evaluation, rollback rate, and extra output-head cost.

This is the most direct way to increase work per causal step without paying 86 host-mediated collectives per token.

### P0-C: forward-port the proven fork in benchmarkable groups

Do not attempt a monolithic rebase. Use current upstream as a clean convergence branch and port in this order:

1. the FA width/padding correctness fix;
2. fused lightning indexer;
3. physical sparse K/V gather, then K reuse/cp.async, then union lists;
4. actual-bound MoE tiling;
5. single-op SwiGLU clamp and fused up+gate MMQ;
6. constant-shape decode;
7. MTP only after ordinary logits match.

After each group, require short-context logits/token equality, a 32K run, a 97K run, VRAM accounting, and a rollback point. PR #25202 should be ported separately as a correctness change if quantized compressed caches are enabled.

### P1-A: fused sqrt-softplus router/top-k

Upstream's generic fused top-k MoE pattern does not match DS4's `sqrt(softplus(logits))` gating chain. A DS4-capable fused router can combine bias, transform, top-6 selection, normalization, and expert weights without materializing every intermediate. It is unlikely to equal the MoE-tile gain, but it is relatively isolated and measurable.

### P1-B: expert parallelism for prefill

Width TP splits every selected expert's 2048 intermediate dimension to 512, which is an inefficient MMQ shape. Expert parallelism would instead shard the 256-expert axis, keep each local expert full-width, route the roughly `ubatch × top6` jobs to owners, and AllReduce the 4096-dimensional partial output.

Expected behavior:

- prefill: reasonably balanced because a 512-token ubatch creates about 3072 expert assignments;
- single-token decode: only six experts are active, so three or four GPUs may work and imbalance is unavoidable;
- communication: still one large output collective per layer, so EP is not automatically good on this no-P2P host.

The experiment is worthwhile only as an A/B against the working width-TP fixture. It should not replace layer split based on utilization alone.

### P1-C: specialized small host collective

NCCL uses shared host memory because CUDA P2P is unavailable. Decode reductions are only 16 KiB. Benchmark three implementations independently of the model:

1. current NCCL AllReduce;
2. pinned-host D2H, AVX2/AVX-512 CPU reduce, H2D broadcast;
3. mapped pinned memory with a persistent GPU/host barrier protocol.

If a custom path cannot materially beat NCCL at 16 KiB over thousands of iterations, stop. If it can, wire it only for the small decode collective; keep NCCL for multi-MiB prefill reductions.

### P1-D: true pipeline throughput, not single-stream latency

For serving multiple requests, measure continuous batching at 1, 2, 4, and 8 active sequences. A layer-pipeline can overlap stages across independent microbatches even though one token from one sequence is serial. Report both aggregate tokens/s and per-request p50/p95 latency. This may be the cheapest way to make all GPUs useful if the real workload admits concurrency.

### P2-A: DS4 context parallelism for 200K+

A DS4-specific long-context design can shard cached positions rather than the one K/V head:

1. lightning-indexer scoring is local to a KV-position shard;
2. merge each shard's local top-k into a global top-k;
3. run sparse FA on local selected positions;
4. combine online-softmax max/sum/output statistics.

Communication then scales with heads/query tiles, not total context length. This is substantially more invasive than width TP and does not solve the 78 GiB expert-weight placement by itself, so it belongs after the MoE and MTP work.

### P2-B: down-expert IQ2_XXS requantization

Changing the routed down experts from Q2_K to IQ2_XXS would save about 144 MiB per layer, roughly 6 GiB for 43 layers, and earlier microbenchmarks show IQ2_XXS is much faster on Ampere. This is a real speed/VRAM lever and would make TP placement easier.

It is also the most quality-sensitive matrix in the expert MLP. The current quant already has a large quality gap to the reference. Do not adopt it without:

- perplexity/KL/top-token agreement against the current GGUF;
- English and Russian reasoning checks;
- tool-call exactness;
- long-context needle retrieval;
- a matched real-prompt speed test.

## Practical decision

Production today should remain the optimized fork in layer mode with MTP. The local TP branch is valuable research infrastructure, not a production replacement.

The next engineering milestone should be defined as one of these measurable outcomes:

- prefill: exceed 500 t/s at the established 97K prompt without reducing output quality;
- decode: exceed 40 t/s shallow and improve accepted tokens/main-eval with adaptive K=2;
- TP/EP: beat layer split on the four-layer fixture before spending another full-model load cycle;
- maintainability: reproduce fork logits and at least 95% of fork speed on current upstream after staged ports.

GPU utilization by itself is not an acceptance criterion.
