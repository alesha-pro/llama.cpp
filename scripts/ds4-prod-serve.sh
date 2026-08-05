#!/usr/bin/env bash
# Production launcher for DeepSeek-V4-Flash (0731 UD-IQ2_M) on 4x RTX 3090,
# with automatic warm-up.
#
#   MODEL=/path/to/model-00001-of-000NN.gguf bash scripts/ds4-prod-serve.sh
#
# Why the warm-up: the first request at any new depth pays a one-time
# allocator climb (the sticky galloc plan and the CUDA pools grow once per
# ubatch) and runs ~3x slower than steady state (~550 vs ~1800+ t/s prefill
# at 350 W). One synthetic full-depth prefill at startup moves that cost to
# server start, so every real request - including the first - runs warm.
# Details: DS4HANDOFF.md section 0e.
#
# Notes:
# - one aligned checkpoint keeps agent rollback prefix reuse fast;
# - every DSV4_* flag below is overridable from the environment;
# - kill switches: DSV4_PREFILL_GRAPHS=0 (prefill CUDA graphs),
#   GGML_CUDA_DISABLE_GRAPHS=1 (all CUDA graphs).
set -u
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/mnt/nvme2/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
BIN=${BIN:-build-v4-cuda/bin/llama-server}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-18080}
CTX=${CTX:-131072}
TS=${TS:-1,1,0.95,1.05}
NGL=${NGL:-999}
FIT_TARGET=${FIT_TARGET:-}
UBATCH=${UBATCH:-384}
BATCH=${BATCH:-8192}
THREADS=${THREADS:-8}
CTX_CHECKPOINTS=${CTX_CHECKPOINTS:-1}
CHECKPOINT_EVERY_NT=${CHECKPOINT_EVERY_NT:--1}
LOG=${LOG:-/tmp/ds4-prod-server.log}
WARM=${WARM:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}

# ship flag set (DS4HANDOFF.md section 5)
export DSV4_CONSTANT_SHAPE=${DSV4_CONSTANT_SHAPE:-1}
export DSV4_SPARSE_FA=${DSV4_SPARSE_FA:-1}
export DSV4_FA_UNION=${DSV4_FA_UNION:-1}
export DSV4_IDX_SKIP=${DSV4_IDX_SKIP:-1}
export DSV4_MOE_TILE=${DSV4_MOE_TILE:-1}
export DSV4_MOE_RESIDENT=${DSV4_MOE_RESIDENT:-1}
export DSV4_GLU_FUSE=${DSV4_GLU_FUSE:-1}
export DSV4_MOE_FUSE=${DSV4_MOE_FUSE:-1}
export DSV4_DECODE_FUSED_IDX=${DSV4_DECODE_FUSED_IDX:-1}
export DSV4_DECODE_RADIX_TOPK=${DSV4_DECODE_RADIX_TOPK:-1}
export DSV4_PREFILL_RADIX_TOPK=${DSV4_PREFILL_RADIX_TOPK:-1}
export DSV4_MMVQ_SMALLK=${DSV4_MMVQ_SMALLK:-1}
export DSV4_STABLE_TOPO=${DSV4_STABLE_TOPO:-1}
export GGML_GALLOC_STICKY=${GGML_GALLOC_STICKY:-1}
export DSV4_PREFILL_GRAPHS=${DSV4_PREFILL_GRAPHS:-1}
export DSV4_AGENT_CKPT_TAIL=${DSV4_AGENT_CKPT_TAIL:-1}

if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "!! port $PORT already in use; stop the old server first: kill \$(lsof -ti :$PORT)" >&2
    exit 1
fi

echo "=== starting llama-server on $HOST:$PORT (log: $LOG) ==="
load_args=(-ngl "$NGL")
if [ "$TS" != "auto" ]; then
    load_args+=(-ts "$TS")
fi
if [ -n "$FIT_TARGET" ]; then
    load_args+=(--fit-target "$FIT_TARGET")
fi

setsid "$BIN" -m "$MODEL" \
    "${load_args[@]}" --split-mode layer --flash-attn on --no-repack \
    --ctx-size "$CTX" --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --ctx-checkpoints "$CTX_CHECKPOINTS" --checkpoint-every-n-tokens "$CHECKPOINT_EVERY_NT" \
    -t "$THREADS" --poll 100 --parallel 1 \
    --host "$HOST" --port "$PORT" --jinja --alias ds4 > "$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 240); do
    kill -0 "$SRV" 2>/dev/null || { echo "!! server died during load:"; tail -20 "$LOG"; exit 1; }
    curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 5
done
curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' || {
    echo "!! server never became ready:"; tail -20 "$LOG"; exit 1; }

if [ "$WARM" != "1" ]; then
    echo "=== server ready (pid $SRV), warm-up skipped (WARM=0) ==="
    exit 0
fi

echo "server ready (pid $SRV) - warming to n_ctx, one-time ~4-5 min"
python3 - "$PORT" "$CTX" <<'EOF'
import json
import sys
import urllib.request

port, n_ctx = sys.argv[1], int(sys.argv[2])
base = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

# The warm pass only exercises shapes, so any text works. Tokenize a big
# synthetic blob and slice to just under n_ctx for an exact-depth prefill.
para = ("The quick brown fox jumps over the lazy dog while seventeen "
        "engineers benchmark a mixture-of-experts transformer on four "
        "consumer graphics cards at increasing context depths. ")
blob = para * (n_ctx * 7 // len(para) + 1)
tokens = post("/tokenize", {"content": blob})["tokens"]
target = n_ctx - 512
if len(tokens) < target:
    print(f"warm blob tokenized short ({len(tokens)} < {target}); warming to that depth")
    target = len(tokens)
r = post("/completion", {"prompt": tokens[:target], "n_predict": 1,
                         "temperature": 0, "cache_prompt": False, "stream": False})
t = r["timings"]
print(f"warm pass: {t['prompt_n']} tokens @ {t['prompt_per_second']:.0f} t/s "
      f"(allocator-climb speed; real requests now run warm)")
EOF

echo "=== production server warm and ready on $HOST:$PORT ==="
