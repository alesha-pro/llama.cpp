# DeepSeek-V4-Flash acceleration — consolidated handoff for research agent

Verified against the live rig on 2026-07-07. Everything below is confirmed to
exist. Hand this whole file to the next agent.

Project: DeepSeek-V4-Flash 284B MoE runs locally on Alexey's 4×RTX 3090 (96 GB,
sm_86, no FP8, PCIe, no NVLink) in an aggressive 2-bit quant, via a patched fork
of llama.cpp with hand-written CUDA kernels. Goal: usable 128-256K context with
fast prefill (target 500-800 t/s, currently 495.43 @97K). Naming rule for any
public text: always "DeepSeek-V4-Flash 284B, 2-bit IQ2_XXS, 87 GB, patched fork"
— never shorten to "runs DeepSeek V4 locally" without the quant (dunk bait).

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
two-GPU pair feasibility test are in `DS4_EXPERT_PARALLEL_2026-07-10.md`.

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
```
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

`ssh alexey@192.168.1.3` (key-based, no password). host llm-server, Ubuntu,
4×RTX 3090 CUDA 12.6. Weights on `/mnt/ssd/models`, engines on `/mnt/ssd/engines`.
Non-interactive ssh does not source .bashrc (pass HF_TOKEN/HF_HOME/uv explicitly).
Kill processes by exact PID (`lsof -ti :18080`), never `pkill -f` (kills own ssh).
BMC if hung: 192.168.1.62 admin/admin, `ipmitool -C 17 chassis power cycle`.

## 4. Launch commands (scripts live on the rig, verified 2026-07-07)

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

Follow-up feasibility result: conventional 4-GPU Lightning sharding is rejected. At decode shape, the 32K scan is 55.27 us and exact Top-512 is 34.82 us; 8K local Top-512 is still 33.65 us, so four local selections plus a merge would be launch-bound and slower. Only a fused persistent score+select design remains plausible, with a sub-1-ms/token ceiling; see `DS4_DECODE_RADIX_TOPK_2026-07-10.md`.
| DSV4_MTP_SPEC=1 + DSV4_MTP_GGUF | MTP speculative decode | decode ×1.2-1.5 |
| DSV4_MTP_EMBD_DEV | (new, 06e8035) place tok_embd mirror device — VRAM valve | — |

Diagnostic-only: DSV4_UNION_STATS, DSV4_IDX_QTILE (for 200K+), DSV4_IDX_F16ACC
(negative), kill-switches DSV4_NO_TOPK_GATHER / DSV4_NO_LIGHTNING_IDX.

## 6. Current numbers (measured on the 4×3090 rig)

```
PREFILL @97K:  342 → 362 (sparse FA) → 436 (MoE-tile) → 443 (fused) → 495.43 t/s (resident MoE)
PREFILL @253K: 256.7 t/s, full 253369-tok prompt in 16.5 min, needle @200K PASS
PREFILL @32K (live agent request): 400.9 t/s
DECODE:        15-18 (raw fork) → 33.49 @8K / 31.23-31.60 @97K fused indexer, no MTP
               current equal-split server: 35.699 @595 ctx; 32.593 baseline / 33.053 radix @129,960 ctx
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

## 8. Further prefill work (the 500 t/s target is now within 0.9%)

- Strategy A: fused-kernel dual accumulation in mmq.cuh (load q8_1 activation
  once, up+gate weights serially, two accumulators, swiglu_clamp epilogue; v1
  without stream-k). Est +4-8%. Highest-effort, hottest kernel.
- Requant down Q2_K → IQ2_XXS: +5-7% near-zero-code (IQ2_XXS 1.74× faster), but
  down is quality-sensitive — needs a perplexity gate + Russian-reasoning check.
  This is Alexey's call.
- External reviews were run via /codex + /opencode (deepseek-v4-pro, kimi, glm)
  through the rig's opencode. codex + glm were most useful on the MTP debug.

## 9. All documentation and content (verified locations)

- **Memory (source-of-truth chronology): `/Users/kts/.claude/projects/-Users-kts-dev-projects-agents-x/memory/ds4-longctx-push.md`** (40.9 KB, detailed, dated). Companion: `alexey-llm-hardware-setup.md`.
- **X thread (RU+EN, the post): `post-queue-ds4-thread.md`** in this repo (agents/x). v3, humanized, hook "×29", 13 tweets + 2 chart placeholders. Posted: https://x.com/superalesha/status/2074569147279724715
- Repo README on GitHub fork (alesha-pro/llama.cpp) has the summary + launch.
- Interactive "path of one request" presentation was published as a claude.ai
  artifact during the session (voxel/inference tour) — link in session history.
- NOTE: earlier consolidated handoff files I referenced in-session
  (ds4-SESSION-HANDOFF.md, ds4-longctx-handoff.md, scratchpad BRIEF-*.md) did
  NOT persist to disk. This file replaces them. The memory file + the GitHub
  commit history + this file are the real, complete record.
