# H5-H7: decode kernel work — sinkhorn register rewrite (+4.1%), fp8kv (+0.2%), MoE RPB (no-op)

Date: 2026-08-05/06. Baseline for all numbers: 42.8 t/s short ctx (H0 config:
PL 270 + lgc 1995), UD-IQ2_M, graphs ON, measured via the prod server
(256-token bursts, temp 0, mean of 3-4).

## H4 (diagnosis, no code): kernel decomposition of the 21 ms chain

nsys with GGML_CUDA_DISABLE_GRAPHS=1 (llama-bench pp128/tg128; graphs-off
makes graph-contained kernels visible to CUPTI — with graphs on they are
invisible in this nsys build, which corrupted an earlier analysis):

- GEMV kernels = ~65% of kernel time. The big attention Q8_0 projections
  (q_b/wo_a/wo_b, ~34 MB each per layer) already run at **~0.77-0.80 TB/s of
  the 0.936 peak** — little headroom.
- MoE expert GEMVs: IQ2_XXS gate/up at **0.52 TB/s**, IQ3_XXS down at
  **0.44 TB/s** (mul_mat_vec_q_moe, 6 experts per launch).
- Two custom per-layer kernels were warp-serial latency bombs at decode
  (1-block launches): `dsv4_hc_split_sinkhorn_f32` 14.4 us x86/token,
  `dsv4_fp8_kv_quantize_f32` 17.3 us x64/token.

Microbenchmarks (in /tmp): empty-kernel graph replay = 0.58 us — the
driver/dispatch is fine; the cost was inside the kernels.

## H5: sinkhorn register-resident rewrite — KEEP (+4.1%)

`dsv4-hc-split-sinkhorn.cu`: the 4x4 comb matrix sinkhorn (20 iterations) now
runs in registers on 4 lanes of one warp (row phases barrier-free, column
reductions via butterfly shuffles), as a compile-time `N_HC=4` template —
runtime-bound loops spilled the matrix to local memory (~1.3 us/iteration).

- Standalone graph-replay bench: 28.8 -> 9.8 us/kernel.
- e2e decode: **42.8 -> 44.54 t/s (+4.1%)**, output coherent.
- Numeric check vs CPU reference: max rel diff 1.1e-6 (butterfly pairing
  changes column-sum order by ~1 ulp).
- Caught and fixed a real bug en route: butterfly must shuffle the running
  accumulator, not the original value (first version produced "DDDD..."
  output; found by the coherence check, fixed, re-verified).

## H6: fp8_kv_quantize two-pass — KEEP (neutral, +0.2%)

`dsv4-fp8-kv-quantize.cu`: 21 block barriers per row -> 2 (per-block maxima
computed into shared memory in pass 1, quantization in pass 2). Bit-exact
same math. e2e 44.54 -> 44.65 t/s — noise-level; kept because it is strictly
less synchronization.

## H7: MoE GEMV rows_per_block sweep — NO-OP (reverted to 2)

DSV4_MOE_GEMV_RPB env sweep (1/2/4/8) on mul_mat_vec_q_moe: 44.6-44.65 t/s
for all values. Block geometry is not the limiter; the per-warp vec_dot
latency structure is. A real improvement needs the ik_llama.cpp-style i-quant
GEMV rewrite (wider loads, more rows in flight per warp) — parked, see below.

## Cumulative decode progress this night

```
day start (PL 220):        39.34 t/s short, 37.61 deep(~130K)
H0 (PL 270 + lgc 1995):    42.8           39.35
H5+H6 (kernels):           44.50 short    40.81 deep   (final validation, same night)
prefill @32K on the final build: 1774 t/s (not hurt; 1675-1761 reference)
```

Net: **decode +13.1% short ctx / +8.5% at 130K depth, prefill intact.**

## Parked follow-ups (ranked)

1. ik_llama.cpp i-quant MMVQ port for IQ2_XXS/IQ3_XXS MoE experts
   (0.44-0.52 -> ~0.75 TB/s potential): +6-8% decode, multi-hour, real risk.
2. quantize_q8_1 dedup: 596 launches/token, many re-quantize the same
   activation for sibling GEMVs: ~+2%.
3. DSpark spec decode (DS4_H1_DSPARK_SPEC_2026-08-05.md): acceptance-capped
   until draft placement/quant improves.
