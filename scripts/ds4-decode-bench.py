#!/usr/bin/env python3
"""Decode-throughput bench against the running DS4 server.

Measures clean decode t/s at two depths:
  - short:  fresh ~100-token prompt, no cache reuse
  - deep:   replays the exact warm-up blob (n_ctx-512 tokens) with
            cache_prompt=true, so prefill is served from the slot cache
            and we decode at ~n_ctx depth.

Usage: python3 scripts/ds4-decode-bench.py [port] [n_ctx] [reps]
"""
import json
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
base = f"http://127.0.0.1:{port}"


def post(path, payload, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


para = ("The quick brown fox jumps over the lazy dog while seventeen "
        "engineers benchmark a mixture-of-experts transformer on four "
        "consumer graphics cards at increasing context depths. ")
blob = para * (n_ctx * 7 // len(para) + 1)
tokens = post("/tokenize", {"content": blob})["tokens"]
deep = tokens[:n_ctx - 512]
short = tokens[:100]

N_PREDICT = 128


def run(prompt, label):
    r = post("/completion", {"prompt": prompt, "n_predict": N_PREDICT,
                             "temperature": 0, "cache_prompt": True,
                             "stream": False})
    t = r["timings"]
    print(f"{label}: prompt_n={t['prompt_n']} (cached {t.get('cached_tokens', '?')}) "
          f"pp={t['prompt_per_second']:.1f} t/s | "
          f"decode {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} t/s "
          f"({t['predicted_ms']/max(t['predicted_n'],1):.2f} ms/tok)", flush=True)
    return t["predicted_per_second"]


results = {}
for label, prompt in (("short", short), ("deep ", deep)):
    vals = []
    for i in range(reps):
        vals.append(run(prompt, f"{label} run{i}"))
    results[label.strip()] = vals

for label, vals in results.items():
    print(f"== {label}: mean {sum(vals)/len(vals):.2f} t/s  "
          f"min {min(vals):.2f}  max {max(vals):.2f}")
