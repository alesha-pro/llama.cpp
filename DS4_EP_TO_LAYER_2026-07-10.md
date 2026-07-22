# DeepSeek-V4 dynamic EP-to-layer transition — 2026-07-10

## Outcome

The private fork can now use expert parallelism for prompt ingestion and
switch once, before the first single-token decode call, to the normal
layer-local expert layout. This combines the faster EP prefill with the faster
layer-split decode without reloading the model or keeping a second worker.

Enable both opt-in paths:

```bash
GGML_CUDA_P2P=1 \
DSV4_EXPERT_PARALLEL=1 \
DSV4_EP_TO_LAYER=1 \
build-v4-cuda/bin/llama-server ... --split-mode layer -ts 1,1,1,0.85
```

The transition is armed only after the context has processed a multi-token
batch. It is one-way for the lifetime of the model. A decode-only process that
never ingests a prompt does not trigger it.

## Implementation

- Retain the four 18,576-MiB EP slabs and expose their simple CUDA buffers.
- Create normal full-expert tensor metadata on the original layer devices.
- Keep the production `-ts 1,1,1,0.85` layer placement; do not move expert
  layers to a different GPU.
- Reuse most of each EP slab as destination storage. Whole tensors that do not
  fit use fragmented 528/672-MiB extensions, avoiding the custom driver's
  failure on a second multi-GiB `cudaMalloc`.
- Treat the old and new layouts as a permutation of 12-MiB blocks. Empty
  extension slots break chains and one 12-MiB device scratch block breaks
  cycles.
- Copy every block with the CUDA backend's direct device/P2P copy path. There
  is no host staging, mmap reload, or H2D transfer.
- Destroy EP scheduler graphs before changing tensor pointers, then reserve a
  fresh ordinary layer graph.

Host reload was explicitly rejected: with the custom NVIDIA 610.43.02 driver,
post-EP H2D copies reproducibly faulted inside `libcuda`, including copies from
small anonymous staging buffers. Sequentially loading a second worker also
cost about 86 seconds and cannot coexist in 4x24-GiB VRAM.

## Full 129,960-token validation

Production model:

`/mnt/nvme/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

Configuration: 4x RTX 3090, custom 610 P2P driver, context 131,072,
ubatch 512, no MTP, `-ts 1,1,1,0.85`, fused decode indexer and exact radix
Top-512 enabled.

| phase | measured result |
|---|---:|
| EP prefill, 129,960 tokens | **500.18 t/s** (259.83 s) |
| device-only layout transition | **4.546 s** |
| physical device copies | 72.598 GiB at 15.97 GiB/s |
| clean decode work, 512 tokens near 130K | **31.12 t/s** (32.13 ms/token) |
| benchmark `t_tg`, including transition | 24.25 t/s |
| total prefill + transition + decode | 280.94 s |

The benchmark's `t_tg` starts before the one-time transition, so it is not the
steady decode rate. A repeated short-context scenario in the same process
measured 29.24 t/s after warmup versus 28.50 t/s for a fresh direct layer-split
control on that run. Thus there was no persistent transition penalty in the
warm short-context A/B.

For the previous long-context layer baseline, prefill took 340.29 seconds at
381.91 t/s and warm decode was about 32.20 t/s. The hybrid saves roughly 76
seconds in prompt ingestion, spends 4.55 seconds once on the transition, and
keeps long-context decode within about 3.4% of that warm baseline. Net time
saved through the first 512 decoded tokens is about 75.3 seconds.

## Operational notes

- This path requires working CUDA peer copies. Keep `GGML_CUDA_P2P=1` and
  verify P2P after kernel/driver changes.
- Use one active sequence (`--parallel 1`). The model-wide layout switch is
  not designed to race with another slot decoding from the EP graph.
- The measured production split needs 2,256 MiB of extension storage on CUDA0
  and CUDA2 and 528 MiB on CUDA1. The full 128K KV allocation passed, but VRAM
  headroom is intentionally tight.
- The 4.55-second transition is paid once per model load, not once per request.
- Do not use this for a server that starts directly from single-token decode
  without first ingesting a prompt; leave EP disabled for that workflow.
