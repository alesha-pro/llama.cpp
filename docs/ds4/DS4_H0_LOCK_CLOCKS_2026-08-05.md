# H0: locked SM clocks (-lgc) — decode +8.8%

Date: 2026-08-05. Rig: 4x RTX 3090, UD-IQ2_M, CTX=131072, prod flags, PL raised 220 -> 270 W.

## Hypothesis

Decode is insensitive to the power *limit* (36.6-37.9 t/s across 220-350 W,
DS4HANDOFF 0e) but the layer-split pipeline leaves each GPU idle most of the
token, so DVFS droops between work windows and ramps back too slowly. Locking
the SM clock should remove the droop. (Hypothesis A5 of
`DS4_DECODE_UPLIFT_RESEARCH_2026-07-10.md`, finally measured.)

## Change

```
sudo nvidia-smi -pl 270      # all 4 GPUs (was 220)
sudo nvidia-smi -lgc 1995    # lock SM clock min=max=1995 MHz
```

No code changes. Memory clock lock (-lmc 9751) tested separately: zero effect.

## Measurements (server /completion, temp 0, n_predict=128, mean of 3-4 runs)

| config | decode short ctx | decode ~130K |
|---|---:|---:|
| PL 270, no lock (baseline) | 39.34 t/s (25.3 ms/tok) | 37.61 t/s (26.5 ms/tok) |
| -lgc 1695 | 39.37 t/s | — |
| **-lgc 1995** | **42.81 t/s (23.35 ms/tok)** | — |
| -lgc 2130 | 42.82 t/s | — |
| -lgc 1995 + -lmc 9751 | 42.78 t/s | — |

- Uplift: **+8.8%** short ctx, stable to ±0.05 t/s across runs.
- Deep ctx (~130K), measured later the same night: **37.61 -> 39.35 t/s
  (+4.6%)**.
- Clock response is binary: 1695 = baseline, 1995 = +8.8%, 2130 = same as 1995.
  The droop removal, not the absolute frequency, is the mechanism.
- Prefill check with the lock active: **1761 t/s @32K fresh prompt** (reference
  1675 t/s same-depth warm) — prefill NOT hurt; under the 270 W cap the power
  governor still throttles when it must, the min-clock request is advisory.

## Caveats

- -lgc needs root and is not persistent across reboots (same as -pl).
- Locked clocks raise idle power draw somewhat; not measured, accepted.
- Deep-context decode with lock to be re-verified at final validation
  (the 130K slot was evicted by the prefill check).

## Verdict

KEEP. Ship config: PL 270 + `-lgc 1995`. New decode baseline: **42.8 t/s short**.
