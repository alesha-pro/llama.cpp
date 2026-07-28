# DS4 research archive

Dated working notes from optimizing DeepSeek-V4-Flash 284B on 4x RTX 3090.
These are a record of what was tried and measured, including the paths that were
rejected — kept because the negative results and their arithmetic are reusable.

The two live documents stay in the repository root:

- **`DS4HANDOFF.md`** — the consolidated handoff: model, rig, launch commands,
  the full `DSV4_*` flag table, and key engine findings.
- **`DS4_OPTIMIZATION_2026-07-27.md`** — the most recent round of changes
  (`DSV4_MMVQ_SMALLK`, `DSV4_PREFILL_RADIX_TOPK`, the tensor-split correction)
  with mechanisms, measurements and the current speed tables.

## Archive, newest first

| file | what it covers |
|---|---|
| `DS4_REAP_K144_2026-07-21.md` | K144 REAP 162B checkpoint: 144-expert/top-6 fused router, GCD-sized EP relayout, 128K and 256K validation |
| `DS4_DECODE_RADIX_TOPK_2026-07-10.md` | exact cooperative radix-select Top-512 for decode; 128K Top-K overlap study; why 4-GPU indexer sharding was rejected |
| `DS4_DECODE_RESEARCH_2026-07-10.md` | decode micro-optimization audit — scheduler copies, fused decode MMVQ, strided reduction, fused down projection; nearly all flat on the real 43-layer model |
| `DS4_DECODE_UPLIFT_RESEARCH_2026-07-10.md` | anatomy of the expert-parallel decode penalty (345 us/layer decomposed) and ranked hypotheses for raising decode |
| `DS4_EP_TO_LAYER_2026-07-10.md` | one-shot device-only EP-to-layer permutation: fast EP prefill, then layer-split decode, no reload |
| `DS4_EXPERT_PARALLEL_2026-07-10.md` | four-way routed-expert sharding: +20.9% prefill, -34.4% decode, and the two-GPU pair feasibility test |
| `DS4_RESEARCH_2026-07-09.md` | fork vs upstream vs Unsloth positioning; classic width tensor parallelism measured and rejected on this rig |

A recurring methodological result worth carrying forward: the four-layer test
fixture systematically misleads. Scheduler overhead dominates it and is
amortized by the real 43-layer model — one change looked like +29.6% on the
fixture and measured -0.11% on the model.
