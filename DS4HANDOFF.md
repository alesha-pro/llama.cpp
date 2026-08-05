# DeepSeek-V4-Flash acceleration — consolidated handoff for research agent

Verified against the live rig on 2026-07-07. Everything below is confirmed to
exist. Hand this whole file to the next agent.

Project: DeepSeek-V4-Flash 284B MoE runs locally on Alexey's 4×RTX 3090 (96 GB,
sm_86, no FP8, PCIe, no NVLink) in an aggressive 2-bit quant, via a patched fork
of llama.cpp with hand-written CUDA kernels. Goal: usable 128-256K context with
fast prefill (target 500-800 t/s, currently 495.43 @97K). Naming rule for any
public text: always "DeepSeek-V4-Flash 284B, 2-bit IQ2_XXS, 87 GB, patched fork"
— never shorten to "runs DeepSeek V4 locally" without the quant (dunk bait).

## 0. K144 REAP 162B result (2026-07-21)

`0xSero/DeepSeek-V4-Flash-162B-GGUF` was downloaded, SHA-256 verified, and
validated on this fork at 128K and 256K. K144 support required a fused
144-expert/top-6 CUDA router case and a dynamic GCD-sized EP relayout unit.

Best hybrid results with `-ts 1,1,1,1`: **500.16 t/s prefill + 31.24 t/s
clean decode at 129,960 tokens**, and **363.97 t/s prefill + 30.08 t/s clean
decode at 260,000 tokens**. The 40.816-GiB one-time EP-to-layer permutation
takes about 2.59 seconds. Peak observed 256K VRAM was only about 14,818 MiB per
GPU. Full commands, compatibility changes, comparisons, and logs are in
`docs/ds4/DS4_REAP_K144_2026-07-21.md`.

## 0a. Latest worktree result — fused decode Lightning Indexer (2026-07-10)

Status: implemented and built locally in the `ds4-longctx` worktree; not yet
committed. The decode change is in `src/models/deepseek4.cpp`.

What changed:

- `DSV4_DECODE_FUSED_IDX=1` switches the single-token indexer scorer from the
  decomposed graph (`mul_mat -> relu -> weighted mul -> permute -> sum_rows`)
  to the existing fused `ggml_lightning_indexer` CUDA op.
- The fused kernel streams each F16 compressed index-cache row once, evaluates
  all 64 indexer heads, applies the same two normalization factors, and writes
  only the reduced score. It avoids materializing and rereading the
  `[n_comp, 1, 64]` F32 score tensor in every ratio-4 attention layer.
- The old decode graph remains unchanged when the flag is absent. Prefill is
  unchanged (it already used the fused Lightning Indexer).

Direct same-build A/B at 7,988 prompt tokens, no MTP, same 64-token greedy
request:

| mode | decode throughput | latency/token | prompt throughput |
|---|---:|---:|---:|
| legacy decomposed decode | 28.74 t/s | 34.79 ms | 518.33 t/s |
| fused decode indexer | 33.49 t/s | 29.86 ms | 514.37 t/s |
| delta | **+16.52%** | **-14.18%** | noise-level |

Correctness: the complete 64-token greedy output was byte-identical between
legacy and fused mode (same content SHA-256
`6c753faf94d0f6253b894c8222fe57675cc9e7d9438fbfe85aa459e75d84db90`).
This context is already beyond the 512 compressed-row top-k threshold, so the
tested path includes the real long-context indexer/gather decode graph.

Long-context result on the full 97,450-token prompt:

- first 64-token run: **31.60 t/s** decode, 494.43 t/s prefill;
- cached repeat over 128 decode tokens: **31.23 t/s**;
- prior stable no-MTP legacy point at approximately 100K was 17.61 t/s, so the
  long-context decode improvement is about **1.79x / +79%** (the legacy point
  is the previously recorded reference, not a fresh symmetric rerun);
- decode now falls only about 6-7% from the direct 8K fused result instead of
  collapsing as the compressed cache grows.

Production flag addition:

```bash
DSV4_DECODE_FUSED_IDX=1
```

`llama-server` and `llama-batched-bench` both build successfully. The active
server log prints `using fused decode Lightning Indexer` once when the decode
graph first enters the new path.

## 0b. Latest worktree result — device-resident MoE scheduler (2026-07-10)

Status: implemented and built locally in the `ds4-longctx` worktree; not yet
committed. Source changes are limited to `ggml/src/ggml-cuda/mmq.cu` and
`ggml/src/ggml-cuda/mmq.cuh`.

What changed:

- `DSV4_MOE_RESIDENT=1` replaces the per-MoE host readback/synchronization with
  a fully device-resident schedule for eligible prefill batches (`ne12 >= 64`).
- A small GPU kernel builds the prefix sum of live token tiles per expert from
  `expert_bounds`; one persistent CTA per SM then claims complete
  `(expert, token tile, output-row tile)` jobs through a device atomic queue.
- Every job owns the full K reduction, so the path needs no stream-k fixup and
  never copies the expert bounds to the CPU.
- The default token tile is 16. It can be overridden with
  `DSV4_MOE_RESIDENT_TILE=N` (clamped to 8..128 and rounded to a multiple of 8).
- When resident mode is active it supersedes the old `DSV4_MOE_TILE` readback;
  the old path remains unchanged when `DSV4_MOE_RESIDENT` is unset.

Authoritative real-prompt A/B, no MTP, same 97,450-token prompt and server
configuration:

| mode | prompt throughput | prompt time |
|---|---:|---:|
| old fused path | 442.14 t/s | 220.40 s |
| device-resident, tile 16 | 495.43 t/s | 196.70 s |
| delta | **+12.05%** | **-23.71 s** |

The resident result repeated at 495.17 and 495.43 t/s (within 0.05%). A clean
512-token DeepSeek-V4 batched benchmark after removing verification code reached
623.18 t/s. A separate 256-expert Qwen canary improved from 2297.66 to 2667.04
t/s (+16.1%). This optimization targets prompt processing; single-token decode
is not expected to improve materially.

Correctness was checked with a temporary deterministic reference kernel using
the same 16-wide MMQ shape. The device schedule and the complete output tensors
matched bit-for-bit:

- Qwen canary: `q4_K` and `q6_K`;
- production DeepSeek V4: `q8_0`, `iq2_xxs`, and `q2_K` on all four GPUs;
- 12 production comparisons total: `schedule=ok`, `mismatches=0`,
  `nonfinite=0`, `max_abs=0`, `max_rel=0`.

The verifier was removed from the clean build after the check. End-to-end text
is not required to be bit-identical to the old scheduler because changing the
MMQ tile shape changes floating-point accumulation order; the scheduler itself
is bit-identical to deterministic MMQ when both use the same tile.

Production flag addition:

```bash
DSV4_MOE_RESIDENT=1
```

No `DSV4_MOE_RESIDENT_TILE` setting is required: the validated value 16 is the
default. `llama-server` and `llama-batched-bench` both build successfully.

## 0c. Full-model expert parallel result (2026-07-10)

`DSV4_EXPERT_PARALLEL=1` shards only the routed expert weights across all four
GPUs while keeping attention, KV, shared experts, and residual work on the
existing layer placement. With P2P enabled, the complete 86.72-GB production
model now loads and runs from the NVMe copy under `/mnt/nvme/ds4-models`.

Fresh same-build full-model A/B at pp512, no MTP:

- layer split mean: **581.08 t/s**;
- four-way expert parallel mean: **702.26 t/s**;
- prefill delta: **+20.86%**;
- tg128 decode: **36.34 -> 23.82 t/s (-34.44%)**.

Decision: keep expert parallel as an opt-in prefill/offline-batch mode. Do not
enable it for the interactive server because the per-layer four-GPU
broadcast/AllReduce dominates single-token decode. Full details and the
two-GPU pair feasibility test are in `docs/ds4/DS4_EXPERT_PARALLEL_2026-07-10.md`.

### Dynamic EP prefill -> layer decode (2026-07-10)

The earlier tradeoff is now removed by the opt-in `DSV4_EP_TO_LAYER=1` path.
With `DSV4_EXPERT_PARALLEL=1 GGML_CUDA_P2P=1`, the process ingests the prompt
in EP layout, performs a one-time device-only 12-MiB-block permutation before
the first decode token, rebuilds the scheduler, and continues in normal layer
layout. No model reload or host staging is used.

Real 129,960-token validation: **500.18 t/s prefill**, **4.546 s transition**,
then **31.12 t/s clean decode near 130K**. The prior layer baseline was 381.91
t/s prefill and about 32.20 t/s warm decode. Net prompt-side saving after the
transition is about **75.9 seconds** (about **75.3 seconds** through 512 decode
tokens). Keep production `-ts 1,1,1,0.85`; an
equal tensor split reduced decode. Full implementation notes and measurements
are in `docs/ds4/DS4_EP_TO_LAYER_2026-07-10.md`.

## 0c2. 2026-07-27 update: +5% decode, and the context ceiling lifted

Two new flags and one launch correction. Full write-up:
**`DS4_OPTIMIZATION_2026-07-27.md`**.

| flag / change | effect |
|---|---|
| `DSV4_MMVQ_SMALLK=1` | both routed expert matmuls sit exactly on the MMVQ `small_k` boundary that a strict `<` excludes; relaxing it to `<=` gives **+5.45% decode @pp512, +4.91% @pp8192**, prefill flat |
| `DSV4_PREFILL_RADIX_TOPK=1` | prefill top-k via `GGML_OP_TOP_K` + a batched one-block-per-row radix-select instead of a full `GGML_OP_ARGSORT`; **removes a hard OOM at 90,112 prompt tokens**, prefill speed unchanged |
| `-ts 1,1,0.90,0.95` | replaces `1,1,1,0.85`; same total footprint, but minimum free VRAM goes **10 MiB -> 684 MiB**, which the MTP decode graph needs at depth |

**MTP and long context are no longer mutually exclusive** — that had been open
since 2026-07-10 not for lack of measurement but because the configuration was
unsatisfiable.

| ctx | prompt | prefill | decode warm (MTP) |
|---:|---:|---:|---:|
| 131072 | 127,356 | 395.93 | **42.648** |
| **163840** | **156,423** | **367.19** | **41.386** |
| 262144 | — | fails at 204,800 | — |

**Max verified context with MTP: 163840.** 262144 fails on a depth-scaled
compute buffer on CUDA3; projected ceiling ~205-220K. MTP is worth +31% at short
context and +20% at depth. The launch command in section 4 below is superseded
by the one in `DS4_OPTIMIZATION_2026-07-27.md`.

## 0d. 2026-08-04: new checkpoint, and prefill turns out to be host-bound

Target changed: **DeepSeek-V4-Flash-0731, Unsloth `UD-IQ2_M`**
(`/mnt/nvme2/models/DeepSeek-V4-Flash-0731-UD-IQ2_M`, 90.9 GB = 84.7 GiB, ~4 GiB
larger than the custom IQ2_XXS build). Two things about it matter:

- **It ships no MTP tensors at all** (43 blocks, zero `nextn`/`eh_proj`), and the
  old MTP GGUF belongs to a different checkpoint, so speculative decode is off
  the table for this quant unless an MTP head is converted for 0731.
- **The expert quants differ**: gate/up are IQ2_XXS (42 layers) + IQ2_S (1),
  down is **IQ3_XXS (41) + MXFP4 (2)** — not Q2_K. The `DSV4_MMVQ_SMALLK`
  arithmetic in section 0c2 was derived for Q2_K down and no longer describes
  this model, though the flag stays harmless.

Baseline reproduced exactly against Alexey's own numbers (32K 426.7/36.3,
98K 411.0/35.6, 131K 384.5/34.5), so all deltas below are honest.

### Two fixes, both about the scheduler rather than the kernels

**1. `GGML_OP_REPEAT` on I32 was running on the CPU backend** (commit `2df82c3`).
The CUDA backend declined REPEAT for I32/I16, so `ggml_backend_sched` placed
every such node on the CPU: 172 nodes per graph, four per layer, from the 4-way
hyper-connections. Each one is a backend boundary — a stream synchronise plus a
D2H and an H2D copy. Decode graph splits **364 -> 20**, CPU splits **176 -> 4**,
matching the ~208 `cudaStreamSynchronize` per token seen in Nsight. The float
`bin_bcast` path is not reusable (it funnels values through `float`, lossy above
2^24), so this adds a dedicated broadcast-copy kernel. `test-backend-ops -o
REPEAT` 14/14. **decode +6.3% @32K, +5.8% @98K**, prefill flat.

**2. A per-ubatch galloc realloc storm serialised prefill** (commit `a21ad15`).
A 32K prefill paid **70 scheduler reallocations, one per ubatch**, and the
realloc path begins by synchronising *all* backends — so the ubatch pipeline
(`sched copies = 4`) never engaged. Two independent causes:

- `DSV4_STABLE_TOPO=1` — `dsv4_pad_fattn_width` emitted its FILL+CONCAT pad pair
  only when the FA width was unaligned, so consecutive 512-token ubatches
  alternated topology (42 node pairs appearing and disappearing) and backend ids
  never matched. An aligned width now pads a full 256 tile under the usual -INF
  mask.
- `GGML_GALLOC_STICKY=1` — the galloc plan is replaced wholesale on every
  reserve, so monotone DSV4 extent growth plus prefill/decode family switching
  re-planned every ubatch. Each `(n_nodes, n_leafs)` family now keeps grow-only
  high-water slot sizes; after one warming pass every later graph of that family
  fits the resident plan. Also records the actual planned block size in
  `hash_node.alloc_size` and frees exactly that (the raw-size free was a latent
  mismatch).

Ship configuration (both flags + `--batch-size 8192 --ctx-checkpoints 0`),
UD-IQ2_M, `--ctx-size 131072`, after one plan-warming request:

| depth | prefill before | prefill after | decode before | decode after |
|---:|---:|---:|---:|---:|
| 32K | 433.8 | **511-521** (+19%) | 35.9 | **39.4** (+9%) |
| 98K | 410.9 | **442** (+7.6%) | 35.4 | **38.3** (+8%) |
| 131K | 384.5 | **404** (+5.0%) | 34.5 | **37.8** (+9.7%) |

The first request at a new depth after a server start still pays the climb
reallocs once; everything after it runs on the resident plan.

### Where prefill actually stands now (the important part)

After the fix, ALL-IDLE fell **51.6% -> 4.1%** and the overlap factor rose
1.00x -> **1.149x** — but each GPU is still only **27-30% busy**. The reason is
no longer the scheduler:

```
520 t/s  <- host thread pinned at 99% CPU
  +- 18.3 s of a 20 s window inside cudaLaunchKernel (121K launches, 151 us each)
       +- perf: >60% of its cycles are the sched_yield syscall machinery
            (libcuda yield-spins waiting for command-channel space;
             the AMD srso mitigation taxes every syscall)
                 +- channels stay full because the work is a serial
                    GPU0->1->2->3 layer chain
```

Ruled out by measurement, all of them plausible beforehand: **P2P** (`GGML_CUDA_P2P`
0 vs 1: 511.2 vs 511.3), **GSP idle-wake** (busy GPUs cost the same to submit as
idle ones), **NVML pollers** (killing nvtop and a polling nvidia-smi: <1%),
**`cudaDeviceScheduleSpin`** (511.9 — spinning waits just as long; patch
reverted), **`--ubatch-size 1024`** (495.8 — launch count scales with tokens,
not with ubatch).

Rig-level microbenchmark, empty kernels, **idle cards**: 3.0 us per launch on
one GPU and the same round-robin across four — so 151 us/launch is contention
under real load, not a driver constant. Caveat learned the hard way: run these
only when the rig is free. A second CUDA process alongside llama-server measures
1270-2520 us/launch and floods dmesg with `GspRmAlloc failed ... status=0x4f` on
GPU1 — channel exhaustion, **not** faulty hardware (the card gives 3 us and zero
errors once the rig is idle).

Diagnostics used are still in the tree behind `GGML_SYNCLOG=1`: `[SYNCLOG]`
counters for realloc / sched-sync / ctx-sync and `[GALLOCFAIL]` lines naming the
first node whose planned size was exceeded.

## 0e. 2026-08-04 late: capture-always CUDA graphs — prefill 513 -> 1162 t/s @32K

The plan-item from section 8 landed the same evening as its recon (commit
`ed7794b`, design "B2"). `DSV4_PREFILL_GRAPHS=1` skips the graph-stability
gate for prefill-sized splits (MUL_MAT_ID with ne2 >= 64) and captures every
ubatch: launches are recorded host-side — never entering the contended
command channels that cost 151 us per launch — then one `cudaGraphLaunch`
submits the whole ~2200-node split. Captures live under a shadow key (key+1)
because prefill and decode share `nodes[0]` and a shared entry would make a
warm decode token silently replay a prefill graph. EndCapture failures retire
the key to direct execution; ExecUpdate failures re-instantiate. The flag is
in the ship set (`ds4m-serve.sh`); kill switch `GGML_CUDA_DISABLE_GRAPHS=1`.

Same-day ladder (0731 UD-IQ2_M, ship flags, batch 8192, srso off). "no B2" is
the same-build control with `DSV4_PREFILL_GRAPHS=0`; B2 numbers are
clean-server warm runs:

| depth | no B2 | B2 warm | gain | decode no B2 / B2 |
|---:|---:|---:|---:|---:|
| 32K | 513.5 | **1484** | ×2.9 | 37.7 / 38.0 |
| 98K | 435.9 | **1460** | ×3.35 | 36.97 / 37.02 |
| 130K | ~444 | **1381** | ×3.1 | — / 36.9 |

That is the section-8 ceiling estimate ("about 3x, implied by summed GPU busy
time") hit almost exactly. During a warm B2 130K prefill the GPUs sit at
**93-96% sm, pinned at the 220 W power cap 94% of the time** — the power
limit is now the binding constraint at depth. Decode is confirmed untouched
(control arm: 36.97 without B2 vs 37.02 with, same day, same build).

Two caveats, both understood and recorded:

- **The first request at a new depth still pays the galloc/pool climb** and
  runs at roughly the old speeds (390 t/s cold at 130K vs 1381 warm). The
  "auto-warm the sticky plan at startup" item in the list below is thereby
  promoted: one synthetic deep pass at server start would make every real
  request run at warm speed.
- **The degraded state is SOLVED** (bisect + fix `a7ff920`): trigger was
  `DSV4_GRAPH_DBG=1`; mechanism was three DSV4GDBG format strings carrying a
  raw embedded newline byte instead of `\n` — emitted lines never terminated,
  the server log path accumulated ever-growing glued lines, and processing
  them throttled the submit thread proportionally to log volume (deep runs
  emit most, hence 98K stuck at 658 while 32K only sagged). Reproduced on
  demand (98K 658/664, no warm recovery), then fixed: the same DBG server now
  runs 98K at **1906 t/s** warm. DBG is safe again; keep it off in prod
  simply because it is diagnostic noise. The first B2 ladder (1162/509/444)
  was measured through this bug and understates B2.

Sanity: 17*23=391 through 3.7K- and 13.7K-token prefills through the capture
path, coherent long output, decode text normal across all arms.

**Agent round-trip validation (same night, `scripts/ds4-agentic-roundtrip.py`).**
The incremental-prefill loop — the pattern none of the day's arms covered —
immediately found a real B2 bug: ragged batch tails produced ever-new shadow
keys, each retired cudaGraphExec kept device memory, and GPU2 OOMed at
cudaGraphLaunch at 120K depth. Fixed in `0e0862d`: one per-device sentinel
key (exactly one graph + instance alive per device) and instantiate/update/
launch failures now fall back to direct execution instead of aborting.
Revalidated worst-case (no checkpoints, full ~130K reprocess per turn, 11
turns to genuine context overflow): no crash, stable VRAM, decode flat,
incremental full-reprocess prefills at 1776-1795 t/s. With default context
checkpoints the loop reprocesses only ~1.4K tokens/turn (the appended chunk)
at stable 38 t/s decode — **prefix reuse works through the capture path**.
Ship note: `--ctx-checkpoints 0` was a bench-purity flag; do NOT pass it when
serving agents — without checkpoints any transcript rollback on this SWA
model degrades to a full re-prefill.

### Agent incremental-prefill alignment fix (2026-08-04, production-safe)

The remaining agent regression was not checkpoint I/O or long-context compute.
The single tail checkpoint originally restored a 31,438-token seed at position
**31,306**, which is not divisible by DS4's 128-token compression ratio. Every
following `ubatch=384` therefore failed `chunk_aligned` and used the enormous
per-token compressor graph. The same 42,735-token context took **1,779 t/s**
when processed from position 0 but only **296 t/s** as an 11,429-token append.

`DSV4_AGENT_CKPT_TAIL=1` now rounds the one useful tail checkpoint backwards
to an `n_swa` boundary (31,232 in that case). At most 127 extra tokens are
replayed, but all full ubatches stay on the batched compressor. Production uses
one checkpoint and disables periodic checkpoints; the full host checkpoint is
intentional. An attempted recurrent/SWA-only on-device checkpoint reached about
1,525 t/s but produced a different greedy answer and was removed.

Correctness-safe `ubatch=384`, `-ts 1,1,0.95,1.05` results from
`scripts/ds4-agentic-roundtrip.py`, after the full-depth startup warm pass:

| turn | approximate context | incremental prefill | wall | decode |
|---:|---:|---:|---:|---:|
| 1 | 44K | **1,475 t/s** | 8.2 s | 39.68 t/s |
| 2 | 54K | **1,543 t/s** | 10.1 s | 38.75 t/s |
| 3 | 64K | **1,483 t/s** | 10.8 s | 38.62 t/s |
| 4 | 75K | **1,292 t/s** | 7.0 s | 38.95 t/s |
| 5 | 85K | **1,354 t/s** | 9.0 s | 38.74 t/s |
| 6 | 95K | **1,242 t/s** | 8.7 s | 38.85 t/s |

The 44K incremental response matched a fresh full recompute byte-for-byte
(SHA-256 `027f97cf93e22d0225d0cfe67cb5b9042c418671c1a808818ad87e17337421d6`).
Authoritative log/result files are `agent-aligned-host-u384.log` and
`agent-aligned-host-u384-turns6.json` under `~/ds4-sweep/results/`.

## 0f. 2026-08-05: REAP K160 193B — self-converted Q3K/Q4K GGUF, quality parity with UD-IQ2_M

`0xSero/DeepSeek-V4-Flash-0731-REAP` (K160: 160 of 256 experts kept, ~193B
params, MXFP4 weights, 107.8 GB) was downloaded to
`/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP` (SHA-256 verified 48/48) and
converted in-house. Product:
`/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP-K160-Q3KQ4K-final.gguf` — 89.9 GB,
1328 tensors. Recipe: routed experts `w1=q3_k, w2=q4_k, w3=q3_k` (down-proj is
the sensitive one), dense tensors Q8_0, compressor APE tensors F32, chat
template embedded from the UD-IQ2_M metadata (REAP repo ships none; the
13,698-char Unsloth DSML thinking template was injected via
`gguf-py/gguf/scripts/gguf_new_metadata.py`, also kept at
`/mnt/ssd/models/ds4-chat-template.jinja`).

Engine support (two commits on `ds4-longctx`): `topk-moe` CUDA `case 160` +
test, and converter changes — `q3_k`/`q4_k` in `DeepseekV4Model` expert quant
aliases and the C `ggml_quantize_chunk` path, APE tensors pinned to F32,
`F8_E8M0 -> uint8` fallback for torch 2.6.

Conversion gotchas, all cost real time:

- The pinned `transformers==5.5.1` in requirements crashes on the deepseek_v4
  config. Convert with an override requirements file: `transformers==4.57.1`,
  `numpy~=1.26.4`, `torch~=2.6.0`, `sentencepiece`, `gguf`, `protobuf<5`.
- Command shape: `LLAMA_CPP_LIBGGML=$PWD/build-v4-cuda/bin/libggml.so uv run
  --no-project --with-requirements <req> python convert_hf_to_gguf.py <model>
  --outtype q8_0 --deepseek4-expert-outtypes w1=q3_k,w2=q4_k,w3=q3_k
  --deepseek4-expert-workers 32 --outfile <out>`.
- APE trap: the 62 tiny `attn/indexer_compressor_ape.weight` tensors must stay
  F32. Quantized to Q8_0 they are rejected by CUDA, the loader silently moves
  them to CPU, and the scheduler shatters the graph into 209 splits instead of
  5 — prefill collapses to ~550 t/s. Diagnose via `GGML_SCHED_DEBUG` and the
  `done_getting_tensors ... cannot be used with preferred buffer type` log
  line. An already-built GGUF can be repaired by rewriting those tensors from
  the safetensors source.

Measured on the rig, warm, 262K ctx: prefill **1,675 t/s @32K**, 1,766-1,771
t/s @64-130K, marginal ~1,830-1,850 t/s; decode 34-36 t/s; VRAM ~92/96 GB.
The full 200+B UD-IQ2_M fits only 128K; K160 is the first config that does
256K with better-than-2-bit weights.

Quality A/B vs UD-IQ2_M (benchlocal-cli, temp 0, canonical runs, JSONs in
`/mnt/ssd/engines/benchlocal-cli/results-*.json`): **total 118/150 vs
117/150 — parity.** medium 67/75 = 67/75, bugfind 14/15 = 14/15, cli-40
23/40 vs 22/40, hermesagent 14/20 = 14/20. Error profiles differ: K160 wins
investigation/multi-step scenarios (CLI-16/17/28/30/40, HA-17/19, BF-09 Go
slice aliasing), UD wins exact-format text tasks (CLI-01/03/11/35, HA-08/13,
BF-15 data race). Russian output does not work on either — accepted.

Known cost (candidate optimization, not started): with
`DSV4_AGENT_CKPT_TAIL=1` at 262K every request pays a fixed ~1.7-3 s for the
single tail checkpoint, which serializes the full memory state — the price
grows with the populated context, so after the warm fill short requests show
~390-800 apparent t/s while long prompts still asymptote to ~1,830. Copying
only the occupied KV range (or checking whether the copy goes via host instead
of P2P) is the obvious next lever.

Prod launch: `MODEL=/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP-K160-Q3KQ4K-final.gguf CTX=262144 bash scripts/ds4-prod-serve.sh`
(the script warms to `n_ctx-512` itself; `EXTRA_ARGS` env hook added for
one-off flags).

## 1. The engine (fork)

- On the rig: `/mnt/ssd/engines/llama.cpp-v4-cchuter`, branch **`ds4-longctx`**.
- Base fork: `cchuter/llama.cpp` (remote `origin`) — the only fork with CUDA
  kernels for V4-Flash at selection time.
- Public: **https://github.com/alesha-pro/llama.cpp** branch `ds4-longctx`
  (remote `ap`, gh logged in as alesha-pro). Default branch = ds4-longctx,
  README has the summary on top.
- Live HEAD (2026-07-07): **`06e8035`**. NOTE: 3 commits past the session that
  wrote most of this (859e602). Work continued: `cfeac53` mirrors tok_embd onto
  the MTP device for candidate lookup (the proper root fix for the MTP
  prefill-boundary crash that 859e602 had gated around), `753e52e` grammar
  robustness, `06e8035` DSV4_MTP_EMBD_DEV VRAM valve.
- Build: `cd /mnt/ssd/engines/llama.cpp-v4-cchuter && cmake --build build-v4-cuda --target llama-server -j 16`.
  Binary: `build-v4-cuda/bin/llama-server`.
- All optimizations are behind `DSV4_*` env flags; flags unset = stock path.

### Commit history = the optimization path (newest first)

Branch `kernel-opt-2026-08-04` carries the two newest commits; everything below
them is on `ds4-longctx`.
```
a21ad15 prefill: stop the per-ubatch galloc realloc storm (DSV4_STABLE_TOPO + GGML_GALLOC_STICKY)  ★
2df82c3 cuda: run GGML_OP_REPEAT on I32/I16 instead of falling back to the CPU  ★
7b09cda docs+tools: agentic round-trip harness; TURING table analysed and deprioritized
06e8035 deepseek4: DSV4_MTP_EMBD_DEV env to place the tok_embd mirror (VRAM valve)
753e52e chat/grammar: survive huge schema bounds, trigger DSML grammar on prefix
cfeac53 deepseek4: mirror token embedding onto MTP device for candidate lookup
859e602 deepseek4: gate MTP candidate branch to genuine decode (fix get_rows OOB at prefill boundary)
9e7a2ba cuda: fused up+gate MMQ path for MUL_MAT_ID + SWIGLU_CLAMP (DSV4_MOE_FUSE=1)
14f9f22 ggml: SWIGLU_CLAMP glu op + DSV4_GLU_FUSE emission for deepseek4 (+0.8% prefill)
169fed1 chat: DSML parser accepts tool parameters in any order  (fixes agent tool calls)
6e8258c server: bypass per-round spec checkpoints for dsv4-mtp (custom rollback)
10f6da6 mmq: per-expert ncols_max for MoE prefill (DSV4_MOE_TILE=1): +19% e2e  ★
efcb133 sparse FA: per-Q-tile UNION lists, ncols1=8 top-k path (DSV4_FA_UNION=1)
3be74cf lightning indexer: causal skip, fp16-acc and q-tiled variants
b1ed25c sparse top-k FA: cp.async gather loads on full tiles (+5.2% @97K)
840610e sparse top-k FA for V4 prompt chunks (DSV4_SPARSE_FA=1): +9% @97K  ★
4600b40 MTP server integration K=1: dsv4-mtp speculative type + rollback ABI fix
8ad0d5e / 44eb8a8 / be4c6b6  MTP phase B (spec decode in-graph, ring-128, rollback)
```
(Earlier commits = MTP phase A + prefill CPU-fallback fix. See memory file.)

## 2. The model

- Prod GGUF: `/mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` (86.7 GB).
  Full 284B, 256 experts / top-8 / ~13B active. Quant: routed experts IQ2_XXS
  (up/gate) + Q2_K (down); attention/shared/output Q8_0; imatrix-calibrated.
- MTP head (speculation): `/mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (3.8 GB).
- Source: antirez/deepseek-v4-gguf. n_vocab 129280. Arch: CSA (lightning indexer
  top-512 + SWA-128) on 22 of 43 layers + dense HCA on 21; head 512; MLA
  (compressed KV ~6.7 MiB/1000 tok).

## 3. Rig access

`ssh <user>@<rig-host>` (key-based; address in the private access note). host llm-server, Ubuntu,
4×RTX 3090 CUDA 12.6. Weights on `/mnt/ssd/models`, engines on `/mnt/ssd/engines`.
Non-interactive ssh does not source .bashrc (pass HF_TOKEN/HF_HOME/uv explicitly).
Kill processes by exact PID (`lsof -ti :18080`), never `pkill -f` (kills own ssh).
BMC if hung (address/credentials in the private access note): `ipmitool -C 17 chassis power cycle`.

## 4. Launch commands

**Current (2026-08-04, 0731 UD-IQ2_M): `bash scripts/ds4-prod-serve.sh`** —
in-repo, self-contained: full ship flag set (incl. capture-always graphs),
binds 0.0.0.0:18080, uses one aligned host checkpoint for agent prefix-reuse
(`DSV4_AGENT_CKPT_TAIL=1`, periodic checkpoints off), and runs an automatic
synthetic full-depth warm pass so the first real request is warm. Defaults are
`--ubatch-size 384` and `-ts 1,1,0.95,1.05`; `WARM=0` skips warming and
everything is overridable from env. The
rig-local `~/ds4-sweep/ds4m-serve.sh` remains the A/B harness launcher, and
`~/ds4-sweep/ds4m-prod.sh` is superseded by the repo script.

### Older checkpoint scripts (on the rig, verified 2026-07-07)

- **`bash /tmp/prod-mtp.sh`** — prod with everything: MTP + all fusions. This is
  the full-fat config.
- **`bash /tmp/prod-nomtp.sh`** — stable fallback without MTP (same prefill, decode ~17@97K).

Explicit full-fat command (what prod-mtp.sh runs):
```bash
cd /mnt/ssd/engines/llama.cpp-v4-cchuter
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DSV4_CONSTANT_SHAPE=1 DSV4_SPARSE_FA=1 DSV4_FA_UNION=1 DSV4_IDX_SKIP=1 \
DSV4_MOE_TILE=1 DSV4_MOE_RESIDENT=1 DSV4_GLU_FUSE=1 DSV4_MOE_FUSE=1 \
DSV4_DECODE_FUSED_IDX=1 \
DSV4_MTP_SPEC=1 DSV4_MTP_GGUF=/mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
./build-v4-cuda/bin/llama-server \
  -m /mnt/ssd/models/DeepSeek-V4-Flash-full-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -ngl 999 --split-mode layer --flash-attn on --no-repack \
  --ctx-size 131072 --batch-size 4096 --ubatch-size 512 -t 8 --poll 100 \
  -ts 1,1,1,0.85 --spec-type dsv4-mtp --parallel 1 \
  --temp 0.7 --top-p 0.9 --repeat-penalty 1.05 --repeat-last-n 256 \
  --host 0.0.0.0 --port 18080 --jinja \
  --reasoning on --reasoning-format deepseek --reasoning-budget 2048
```
Endpoint: `localhost:18080`, OpenAI-compatible `/v1`. Server may be DOWN now
(bring up with the script). NOTE: a vLLM instance sometimes occupies GPU0 (20 GB)
and blocks the load — check `nvidia-smi` first, free GPU0 if needed.

Do-not-touch parameters: `--reasoning-budget 2048` (else it reasons forever and
never answers), `--repeat-penalty 1.05` (1.2 breaks reasoning), `--ubatch-size 512`
(1024 slower, 2048 crashes graph_reserve), `-ts 1,1,1,0.85` (frees GPU3 for MTP weights).

## 5. Env flags (all optimizations)

| Flag | Effect | Gain |
|---|---|---|
| DSV4_CONSTANT_SHAPE=1 | depth-bucketed decode shapes → CUDA-graph replay | base |
| DSV4_SPARSE_FA=1 | sparse top-k FlashAttention for prompt chunks | +9% |
| DSV4_FA_UNION=1 | per-Q-tile union top-k lists | +1.3% |
| DSV4_IDX_SKIP=1 | causal-skip in lightning indexer | free |
| DSV4_MOE_TILE=1 | mmq tile sized to real per-expert token count (was 87% dead MACs) | +19% |
| DSV4_MOE_RESIDENT=1 | persistent GPU work queue; removes expert-bound D2H readback/sync | +12.05% over the previous fused 97K path |
| DSV4_GLU_FUSE=1 | SWIGLU_CLAMP glu op (unblocks matmul fusion) | +0.8% |
| DSV4_MOE_FUSE=1 | fused up+gate MMQ (shared quantize/ids/bounds) | +0.7% |
| DSV4_DECODE_FUSED_IDX=1 | fused single-token Lightning Indexer scoring | +16.5% @8K; about +79% @97K vs the prior long-context reference |
| DSV4_DECODE_RADIX_TOPK=1 | exact cooperative radix-select Top-512 instead of full 32K argsort | 32.593 -> 33.053 raw t/s @129,960 ctx cold (+1.41%); 33.804 warmed sample |
| DSV4_MMVQ_SMALLK=1 | relax the MMVQ `small_k` trigger from `<` to `<=`; both routed expert matmuls (IQ2_XXS up/gate and Q2_K down) sit exactly on the boundary | **+5.45% decode @pp512, +4.91% @pp8192**, prefill flat (n=3/arm, non-overlapping ranges). See `DS4_UPSTREAM_KERNEL_PORT_2026-07-27.md` |
| DSV4_PREFILL_RADIX_TOPK=1 | prefill indexer top-k via `GGML_OP_TOP_K` + a batched one-block-per-row radix-select, instead of a full `GGML_OP_ARGSORT` | **removes the 90K prefill OOM ceiling** — `-ts 1,1,1,0.85` now completes 104,868 tokens. Prefill speed neutral (+0.5%). See `DS4_PREFILL_RADIX_TOPK_2026-07-27.md` |

Follow-up feasibility result: conventional 4-GPU Lightning sharding is rejected. At decode shape, the 32K scan is 55.27 us and exact Top-512 is 34.82 us; 8K local Top-512 is still 33.65 us, so four local selections plus a merge would be launch-bound and slower. Only a fused persistent score+select design remains plausible, with a sub-1-ms/token ceiling; see `docs/ds4/DS4_DECODE_RADIX_TOPK_2026-07-10.md`.
| DSV4_MTP_SPEC=1 + DSV4_MTP_GGUF | MTP speculative decode | decode ×1.2-1.5 |
| DSV4_MTP_EMBD_DEV | (new, 06e8035) place tok_embd mirror device — VRAM valve | — |

| DSV4_STABLE_TOPO=1 | (2026-08-04) constant prefill graph topology: pad the FA width even when already aligned | part of prefill +19% |
| GGML_GALLOC_STICKY=1 | (2026-08-04) grow-only galloc plan per graph family; kills the per-ubatch realloc + all-GPU drain | part of prefill +19% |
| DSV4_AGENT_CKPT_TAIL=1 | (2026-08-04) keep one tail checkpoint aligned to the 128-token DS4 compressor boundary | 296 -> 1,475 t/s on the first 10K agent append; correctness-safe host state |
| GGML_SYNCLOG=1 | (2026-08-04) diagnostic: [SYNCLOG] realloc/sched-sync/ctx-sync counters + [GALLOCFAIL] node that overflowed the plan | — |
| DSV4_RESERVE_FULL=1 | (2026-08-04) reserve the deep-prefill plan at n_ctx - n_ubatch; superseded by the sticky plan | neutral |
| GGML_GALLOC_PAD=1 | (2026-08-04) grid-round planned sizes (max +19% headroom); partial, superseded by sticky | partial |

Diagnostic-only: DSV4_UNION_STATS, kill-switches DSV4_NO_TOPK_GATHER /
DSV4_NO_LIGHTNING_IDX.

Measured **negative on 0731 UD-IQ2_M** (32K, control arm 420.4): DSV4_IDX_QTILE
-4.0%, DSV4_IDX_F16ACC -4.8%, both together -8.7%, DSV4_MOE_RESIDENT_TILE=8
-13.1% (tile 32 was +2.4%, inside the ~3% between-restart noise), ubatch 1024
-3.4%, ubatch 2048 aborts in graph_reserve. The lightning indexer is close to
the hardware ceiling here — with causal skip it runs at roughly 37 TFLOPS
against the 71 TFLOPS FP16-with-FP32-accumulate limit of a 3090, which is why
its alternative kernels cannot win.

Expert parallel re-measured on this checkpoint: `DSV4_EXPERT_PARALLEL=1` gives
**+22% prefill** (531 vs 434 @32K, 501 vs 411 @98K) for **-32% decode** — an
honest trade, not a free win. `DSV4_EP_TO_LAYER=1` speeds up only the *first*
ingest (529 t/s), then reverts to layer-split rates, so it is a one-off saving
of ~70 s on a 130K ingest.

## 6. Current numbers (measured on the 4×3090 rig)

```
PREFILL @97K:  342 → 362 (sparse FA) → 436 (MoE-tile) → 443 (fused) → 495.43 t/s (resident MoE)
PREFILL @253K: 256.7 t/s, full 253369-tok prompt in 16.5 min, needle @200K PASS
PREFILL @32K (live agent request): 400.9 t/s
DECODE:        15-18 (raw fork) → 33.49 @8K / 31.23-31.60 @97K fused indexer, no MTP
               current equal-split server: 35.699 @595 ctx; 32.593 baseline / 33.053 radix @129,960 ctx
```

2026-08-04, **0731 UD-IQ2_M** (different checkpoint and quant mix — not
comparable to the rows above), ship flags + batch 8192 + `--ctx-checkpoints 0`,
no MTP, after one plan-warming request:

```
32K:   511-521 t/s prefill, 39.4-39.5 t/s decode
98K:   442 t/s prefill,     38.3 t/s decode
131K:  404 t/s prefill,     37.8 t/s decode
GPU utilisation during prefill: 27-30% per card - host-bound on kernel submission
```

Same day, after srso/retbleed off (+1.9% @32K) and capture-always CUDA graphs
(section 0e):

```
At the long-standing 220 W power limit:
32K:   1484 t/s prefill, 38.0 decode    (513.5 same-day no-B2 control)
98K:   1460 t/s prefill, 37.0 decode    (435.9 same-day no-B2 control)
130K:  1381 t/s prefill, 36.9 decode
GPU utilisation during warm prefill: 93-96%, power-capped at 220 W 94% of time

After nvidia-smi -pl 350 (same session, warm 130K, power curve):
220 W: 1381   280 W: 1679   300 W: 1742   350 W: 1807-1823
32K @350 W: 1862 t/s. Decode power-insensitive (36.6-37.9 across limits).
Real draw at -pl 350: 307-331 W mean, temps max 83 C in a 3-min burst.
Cold first request per depth still climbs slowly (390 @220 W, 510 @350 W)
Power limit is NOT persistent across reboots (something sets 220 at boot).
```
Historic headline: CPU-fallback fix took prefill 31.6 → 531 t/s (a 23.5K prompt
from 25 min to 53 sec). This is the "×29" hook of the X thread.

## 7. Key engine findings (the substance for a research agent)

1. CPU-fallback: FA CUDA kernel needs KV width % 256; chunks missed it → kernel
   silently fell back to CPU (GPU idle 93%). Fix = pad KV to /256 + mask. ×16.8.
2. MoE 87% dead MACs: 512-token batch → ~16 tokens/expert, but mmq tiled for 4096
   → 128-wide tiles multiplying zeros. Fix = read real max from expert_bounds.
   +19%. Bonus: IQ2_XXS 1.74× faster than Q2_K in MMQ.
3. SWIGLU_CLAMP: swiglu_clamp_exp=10 on all layers meant clamp expanded to 4
   elementwise ops, so the {MUL_MAT_ID,MUL_MAT_ID,GLU} fusion pattern never
   formed. New GLU op collapses it → unblocks fused up+gate.
4. Tool calls: model emits perfect DSML, server PEG parser required params in
   schema order → dropped valid calls. Fix = any-order (169fed1). Also: agent
   harness skills (e.g. superpowers) inject a non-DSML tool format that derails
   the model — disable extra skills for this model. Use temperature 0 for code.
5. MTP crash at depth: candidate embed = get_rows(tok_embd, argmax) crossed
   CUDA→CPU with a bad i32 copy at the prefill boundary. Session fix = gate to
   real decode (859e602); proper fix later = mirror tok_embd onto MTP device
   (cfeac53) + DSV4_MTP_EMBD_DEV (06e8035).

## 8. Further prefill work (next: CUDA graphs for prefill ubatches)

Superseding the older plan below: as of 2026-08-04 prefill is **host-bound on
kernel submission** (section 0d), so kernel-level work cannot move it. The GPUs
are 27-30% busy and the host thread spends its cycles yield-spinning inside
cudaLaunchKernel. Ranked by expected value:

- **CUDA graphs for prefill ubatches — DONE 2026-08-04 (section 0e).** Landed
  as capture-always (design B2), not bucketed replay (B1): +126% @32K to
  1162 t/s, +15% @98K, +10% @130K. B1 (stability program: sanitized props,
  slot keys, ARANGE-as-data, SET_ROWS cache writes, CS_BUCKET-style width
  bucketing) remains the phase-2 option if profiling still shows host-bound
  sections at depth — but the next depth wins more likely come from the
  follow-ups below.
- **`spec_rstack_overflow=off` — DONE 2026-08-04, kept.** Rebooted with
  `spec_rstack_overflow=off retbleed=off` (EPYC 7642 is Zen 2, so the retbleed
  return thunk had to go too; STIBP also relaxed always-on → conditional).
  Syscall cost collapsed — sched_yield 549 → 162 ns, getpid 195 → 69 ns, vDSO
  control unchanged — but e2e moved only 503.8 → 513.5 t/s prefill @32K
  (+1.9%, non-overlapping 3-run warm ranges), decode flat: the spin waits on
  GPU channel drain, so cheaper syscalls only cut detection latency. Stamped
  microbench + rerun script: `~/ds4-sweep/srso/` (results.txt, run.sh,
  after-reboot.sh); harness arms `srso-arm0-32k` / `srso-arm2-32k`. Rollback:
  restore `/etc/default/grub.bak-2026-08-04` + update-grub.
- `-ts` rebalancing and raising the 220 W power limit are **not** useful yet:
  at 27-30% utilisation the cards are not the constraint (they draw ~175 W of
  220 under load). Revisit both once the launch path is fixed.
- **Multi-threaded submission** — the alternative (or complement) to graphs, if
  capture turns out to be blocked by DSV4's dynamic shapes. Today one host
  thread submits for all four devices in sequence at 99% CPU; a submitter thread
  per device would overlap the yield-spin instead of serialising it.
  `ggml_backend_sched` is single-threaded here, so this is the more invasive of
  the two — try graphs first.
- **The last 4 CPU splits**: `GET_ROWS` over `token_embd.weight` (347 MB, kept
  on the CPU) still runs on the CPU backend once per graph. Tiny next to what
  was fixed, but it is now the only remaining backend boundary.
- **Auto-warm the sticky plan at startup** — the one blemish of
  `GGML_GALLOC_STICKY` is that the first request at a new depth after a server
  start still pays the climb. A synthetic warm pass at `n_ctx` during
  `sched_reserve` would hide it; needs care not to inflate the reserve buffers.
- Older ideas, still valid but smaller: fused-kernel dual accumulation in
  mmq.cuh (est +4-8%), and requanting down to IQ2_XXS (+5-7%, quality-sensitive,
  needs a perplexity gate — and note this model's down is already IQ3_XXS).
- External reviews were run via /codex + /opencode (deepseek-v4-pro, kimi, glm)
  through the rig's opencode. codex + glm were most useful on the MTP debug.

### CUDA-graphs recon for prefill (2026-08-04 evening — done, implementation next)

Instrumented and measured at 32K (arm `pg-recon-32k`: 514.1 t/s = baseline,
patch inert). Two flag-gated changes live uncommitted on
`kernel-opt-2026-08-04`: `DSV4_PREFILL_GRAPHS=1` lifts the MUL_MAT_ID capture
veto when the resident MoE route handles the node (mmq, sync-free), and
`DSV4_GRAPH_DBG_FULL=1` prints a field-level diff of every node whose cached
graph properties changed.

Facts established:

- prefill runs as only **4 CUDA splits** (one per GPU, 2157-2407 nodes each);
  the graph keys are stable across ubatches (STABLE_TOPO + STICKY keep the
  arena addresses fixed), so the keyed-graph map does not fragment;
- with the veto lifted nothing else fails compatibility — but capture never
  engages because node properties change **every ubatch** and warmup
  (2 stable calls) never completes;
- the churn by field (484 diff lines from one 32K prefill):
  (a) sched pipeline `copies=4` rotates the split inputs (pos, kq_mask, row
      indices, hidden state) → `sX.data` changes with period 4;
  (b) ARANGE nodes bake n_past into `op_params` (3 per CSA layer);
  (c) compressed-cache writes are CPY-into-VIEW at offset n_past →
      `data`/`view_offs` walk (main KV uses SET_ROWS and is clean);
  (d) every `[n_comp]`-shaped indexer/score tensor grows `ne`/`nb` per ubatch
      (heavy ubatches: ~2000 of ~2200 nodes changed);
- prefill (n=2231) and decode (n=2198) **collide on the same graph key** (same
  first-node arena address) and thrash each other's warmup state;
- the DSV4 kernels are already capture-aware: MOE_TILE readback skips itself
  under capture, the resident path never syncs, union-FA falls back to
  per-token under capture (its 4-byte D2H+sync per call is why it cannot be
  captured as-is; fattn.cu:206-217).

Two viable designs, in build order:

1. **B2 "capture-always"** (recommended first; backend-only). For batch>=64
   splits skip the stability gate: BeginCapture every ubatch, execute into the
   graph (launches build host-side — no command-channel submission, no 151 us
   yield-spins), EndCapture + instantiate (~0.1-0.3 ms at 2.2K nodes) + one
   cudaGraphLaunch. Per-split submission drops from ~65 ms (~430 launches at
   151 us contention) to ~1-2 ms. No model-graph changes: ARANGE, views and
   slot rotation are captured fresh each ubatch. Technical risk: **pool growth
   during capture** (per-ubatch alloc sizes grow) — needs pool-size grid
   rounding plus a capture-abort→direct fallback for growth ubatches.
2. **B1 "stability program"** (classic replay; model surgery). Sanitize the
   property comparison (drop host struct-pointer noise), key on (nodes[0],
   n_nodes, copy slot), replace prefill ARANGE with data-driven position
   inputs, convert compressed-cache writes to SET_ROWS, bucket prefill widths
   to DSV4_CS_BUCKET (2048 comp rows = 8192 tokens) exactly like decode
   already does, relax warmup to one visit. Replay fraction 50-87% by bucket.
   Phase 2 if B2 leaves GPU-busy headroom.

### Open experiments (cheap, not yet run)

- **Raise the 220 W power limit — DONE, curve measured** (warm 130K):
  220 W 1381 / 280 W 1679 / 300 W 1742 / 350 W 1823 t/s. 280 W captures
  two-thirds of the full-watts gain; 350 W ran at 83 C max in a 3-minute
  burst (GDDR6X junction temp not exposed by this driver — watch it if 350 W
  becomes the sustained default). Limit resets at boot; persistence needs
  whatever service currently sets 220 to be updated.
- **Auto-warm the sticky plan + pool at startup** — promoted by 0e: the first
  request per depth runs ~3.5x slower than warm (390 vs 1381 @130K). One
  synthetic full-depth pass during server start hides the climb from every
  real request.
- ~~Bisect the DSV4_GRAPH_DBG degraded state~~ — DONE, fixed (`a7ff920`,
  details in 0e): unterminated DBG log lines throttled the submit thread.
- **Capture-safe union-FA** — under capture the union path self-disables (its
  max_union overflow check is a 4-byte D2H + sync, fattn.cu). Was estimated
  from the old +1.3%; with the launch storm gone its true share should be
  re-measured before investing.
- **`-ts` rebalancing** — live again now that the GPUs are busy; interacts
  with the power-limit change, so sweep after it.
- **Expert parallel together with the 2026-08-04 fixes** — never measured in
  combination. EP alone is +22% prefill / -32% decode; the fixes lift both. If
  the decode penalty stays proportional this is still a prefill-only mode, but
  the arithmetic changes and it is one server start to find out.
- **Maximum context on this checkpoint** — the old 163840 ceiling was measured
  *with* MTP weights resident on CUDA3. UD-IQ2_M has no MTP head, which frees
  roughly 5 GB, so the ceiling should be higher. Untested.
- **`DSV4_MOE_RESIDENT_TILE=32`** measured +2.4% at 32K, inside the ~3%
  between-restart noise. Needs 3+ repeats per arm to call; tile 8 is clearly bad
  (-13.1%), so the curve is worth one careful sweep.
- **Depth beyond 131K on the ship config** — 32K/98K/131K are covered, 163K+ is
  not.

### Parked by decision (recon done, so it is cheap to restart)

**MTP head for 0731.** Alexey's call was to skip it this session, but the
groundwork is done and worth not re-deriving: the MTP tensors live in shards
46-48 of `deepseek-ai/DeepSeek-V4-Flash-0731` (4705 tensors, ~10.9 GB, so a
partial download is enough), `convert_hf_to_gguf.py` in this fork already knows
`model.mtp_block.N` / `mtp_emb_norm` / `skip_mtp`, and the runtime loader
(`dsv4_mtp_get` in `src/models/deepseek4.cpp`) accepts *any* GGUF whose tensors
are named `mtp.0.*` — it just reads them onto `DSV4_MTP_DEV`. The work is the
name mapping (HF `mtp.0.attn.wq_a.weight` + `.scale` → the fork's flat names),
the FP8 scale handling, and merging 256 per-expert tensors. On the previous
checkpoint MTP was worth +31% short / +20% deep decode.

## 9. All documentation and content (verified locations)

**Measurement harness (2026-08-04, on the rig at `~/ds4-sweep/`)** — reuse it
rather than rebuilding one:

- `ab2.sh` — one A/B arm: kills the previous server by PID, waits for real
  readiness, sweeps, verifies the flag actually engaged, tears down. `SWEEP=0`
  leaves the server up for interactive work.
- `ds4m-serve.sh` — the UD-IQ2_M launch script; every flag overridable from the
  environment, ship defaults already set.
- `ds4bench2.py` + `prompts/p{32768,65536,98304,130560}.txt` — depth sweep over
  pre-cut prompts, so every arm sees byte-identical input and no arm depends on
  `/tokenize`. `cutprompts.py` regenerates them.
- `overlap.py` — per-GPU busy, ANY-BUSY, ALL-IDLE and the overlap factor from an
  nsys sqlite; `kerncount.py` — launch counts (the fusion-target list);
  `diffprof.py` / `diffapi.py` — differential kernel and API profiles between
  two traces; `timeline.py` — chronology of one step.
- `summary.py` — collects every arm in `results/` into one delta table.
- `req.py` — single request with timings; `sanity.py` — three prompts with
  checkable answers (17*23=391), the guard against a kernel that passes a shape
  test but corrupts real generation.


- **Memory (source-of-truth chronology): `/Users/kts/.claude/projects/-Users-kts-dev-projects-agents-x/memory/ds4-longctx-push.md`** (40.9 KB, detailed, dated). Companion: `alexey-llm-hardware-setup.md`.
- **X thread (RU+EN, the post): `post-queue-ds4-thread.md`** in this repo (agents/x). v3, humanized, hook "×29", 13 tweets + 2 chart placeholders. Posted: https://x.com/superalesha/status/2074569147279724715
- Repo README on GitHub fork (alesha-pro/llama.cpp) has the summary + launch.
- Interactive "path of one request" presentation was published as a claude.ai
  artifact during the session (voxel/inference tour) — link in session history.
- NOTE: earlier consolidated handoff files I referenced in-session
  (ds4-SESSION-HANDOFF.md, ds4-longctx-handoff.md, scratchpad BRIEF-*.md) did
  NOT persist to disk. This file replaces them. The memory file + the GitHub
  commit history + this file are the real, complete record.
