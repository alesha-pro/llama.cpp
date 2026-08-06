# H1: DSpark speculative decode for 0731 — built, validated, NOT shippable yet

Date: 2026-08-05. Status: working implementation, net-negative e2e at the
current VRAM/acceptance point. Everything is behind env flags; default off.

## What was built

1. **Draft weights**: the official 0731 DSpark module (3 chained MoE blocks
   `mtp.0/1/2`, 160 pruned experts each) exists in the local REAP checkpoint
   (`/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP`, shards 46-48). Converter
   `scripts/ds4-convert-mtp.py` (table-driven, reuses the K160 conversion
   helpers) emits GGUFs:
   - `DeepSeek-V4-Flash-0731-REAP-DSPARK-Q4K-Q8_0.gguf` (6.91 GiB, experts Q4_K)
   - `...-Q3K-Q8_0.gguf` (5.42 GiB), `...-Q2K-Q8_0.gguf` (3.6 GiB)
   All numerically verified against the source safetensors (cossim 1.0/0.998).
2. **Graph** (`src/models/deepseek4.cpp`): the old single-block MTP branch
   (e_proj/h_proj, gone in 0731) was replaced by the DSpark architecture:
   - taps: HC-mean of layers 40-42 outputs -> `main_x = main_norm(main_proj(concat))`;
   - ingest: per-block ring KV = fp8(rope(kv_norm(wkv_b(main_x)))) — 3 rings;
   - candidate: embeddings broadcast over HC channels -> 3 chained blocks ->
     head (mtp.2 hc_head + norm + shared output) + **markov bigram bias**
     (`markov_w2 @ markov_w1[tok]` added to draft logits, as in the official
     `forward_head`).
   - K=2 chained draft (`nc>=3`) disabled — needs redesign for the chain.
3. **Offline acceptance tool**: `llama-ds4-mtp-accept` (DSV4_ACCEPT_DBG=1 dumps
   intermediates; DSV4_MTP_NO_MARKOV / DSV4_MTP_MARKOV_ONLY ablations).

## Measured acceptance (teacher-forced, greedy, docs/code text)

| draft | acceptance |
|---|---:|
| 3-block chain, no markov | 7.0% |
| markov bias only, no blocks | 2.6% |
| **3 blocks + markov (Q4_K)** | **27.5% docs / 26.8% code** |
| 3 blocks + markov (Q2_K) | 19.5% docs |
| live server (Q2_K, 32K ctx) | 13-21% per request |

The chain and the markov head are synergistic — neither works alone.

## Live e2e (Q2_K draft, ctx 32768)

- spec ON: 17.8-19.4 t/s — **2.3x SLOWER than baseline**.
- spec OFF, same config (TS tilt + expert overrides): 41.0 t/s (vs 42.8 clean
  baseline: the overrides cost only ~4%).
- Output text verified coherent (verify/rollback machinery works correctly).

Root cause of the negative: the draft adds ~30 ms/token — 3 MoE blocks +
full-vocab head per candidate (x2 candidates) run as uncaptured loose kernels
plus cross-device hops, while acceptance only multiplies ~x1.2.

## VRAM fit (the hard part, solved)

UD-IQ2_M at 131K ctx leaves only 2.0/1.8/1.2/3.1 GiB free per GPU. Working
recipe (ctx 32768): `TS=0.86,1.00,1.00,1.14` (layer boundaries via
`upper_bound(cum_ts, il/44)` — deterministic once known), `-ot blk.42.*_exps=CUDA0`,
`-ot blk.41.{up,down}_exps=CUDA1`, `DSV4_MTP_EMBD_DEV=CUDA1`, Q2_K draft.

## What's needed to make it a win (next steps, in order)

1. Draft ops captured in CUDA graphs (the draft branch currently runs loose).
2. Single-device draft placement (needs ~4 GB free on one GPU -> probably a
   smaller main quant or K160 base, which is REAP-native and may also raise
   acceptance).
3. Q4_K draft (acceptance 27.5% vs ~20% at Q2_K) once VRAM allows.
4. Only then: K>=2 with block_size>1 noise-token drafting (official drafts 5
   per stage), confidence head for adaptive depth.

## Verdict

Parked, code intact behind `DSV4_MTP_SPEC`/`DSV4_MTP_GGUF`. The night's
shippable decode win remains H0 (locked clocks, +8.8%).
