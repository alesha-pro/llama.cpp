#!/usr/bin/env python3
"""Profile the agentic round-trip against a running DS4 llama-server.

Everything else in DS4_OPTIMIZATION_2026-07-27.md measures one large prefill plus
a decode burst. A coding agent does something different: ingest a large context
once, then loop {append a tool result, decode a response} tens of times against a
monotonically growing cache. This measures that loop.

What it answers:
  1. Is the cached prefix actually reused at depth, or is part of it re-prefilled?
     (llama-server reports prompt_n = tokens it actually had to process.)
  2. Per-turn latency split: incremental prefill vs decode, at increasing depth.
  3. Whether small incremental prefills stay on the fast MoE paths.

Usage:
  python3 scripts/ds4-agentic-roundtrip.py --base http://127.0.0.1:18080 \
      --seed-file <big.txt> --turns 8 --tool-result-tokens 1200 --decode 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request


def post(base: str, path: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def completion(base: str, prompt: str, n_predict: int, timeout: int = 3600):
    t0 = time.time()
    r = post(base, "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": True,
        "stream": False,
    }, timeout=timeout)
    return r, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080")
    ap.add_argument("--seed-file", required=True,
                    help="text file used as the initial large context")
    ap.add_argument("--seed-bytes", type=int, default=100_000)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--tool-result-tokens", type=int, default=1200,
                    help="approximate; converted at ~3.1 bytes/token")
    ap.add_argument("--decode", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    filler = open(args.seed_file, errors="replace").read()
    if len(filler) < args.seed_bytes:
        filler = filler * (args.seed_bytes // max(1, len(filler)) + 1)
    context = filler[: args.seed_bytes]

    # A tool result is just more text appended to the transcript. Slice it from a
    # different offset each turn so the tokens are genuinely new, not a repeat the
    # cache could collapse.
    chunk_bytes = int(args.tool_result_tokens * 3.1)

    rows = []

    r, wall = completion(args.base, context, 8)
    t = r.get("timings", {}) or {}
    depth0 = t.get("prompt_n", 0)
    print(f"initial ingest: prompt_n={depth0} "
          f"pp={t.get('prompt_per_second', 0):.1f} t/s  wall={wall:.1f}s", flush=True)
    rows.append({"turn": 0, "kind": "ingest", "prompt_n": depth0,
                 "pp": t.get("prompt_per_second"), "tg": t.get("predicted_per_second"),
                 "wall": wall})

    print(f"\n{'turn':>4} {'depth':>8} {'new_tok':>8} {'reproc':>7} "
          f"{'pp t/s':>9} {'tg t/s':>8} {'wall s':>7}", flush=True)

    for i in range(1, args.turns + 1):
        off = (len(filler) // 2 + i * chunk_bytes) % max(1, len(filler) - chunk_bytes)
        context += "\n\n<tool_result>\n" + filler[off: off + chunk_bytes] + "\n</tool_result>\n"
        r, wall = completion(args.base, context, args.decode)
        t = r.get("timings", {}) or {}
        # prompt_n is what the server actually had to process this turn. If the
        # cache is doing its job it is ~the appended chunk, not the whole context.
        reproc = t.get("prompt_n", 0)
        rows.append({"turn": i, "kind": "turn", "prompt_n": reproc,
                     "pp": t.get("prompt_per_second"), "tg": t.get("predicted_per_second"),
                     "predicted_n": t.get("predicted_n"), "wall": wall,
                     "approx_ctx_bytes": len(context)})
        print(f"{i:>4} {len(context)//3:>8} {chunk_bytes//3:>8} {reproc:>7} "
              f"{(t.get('prompt_per_second') or 0):>9.1f} "
              f"{(t.get('predicted_per_second') or 0):>8.2f} {wall:>7.1f}", flush=True)

    turns = [r for r in rows if r["kind"] == "turn"]
    if turns:
        tg = [r["tg"] for r in turns if r.get("tg")]
        print("\n--- summary ---")
        print(f"turns              : {len(turns)}")
        print(f"reprocessed/turn   : median {statistics.median(r['prompt_n'] for r in turns):.0f} tokens")
        if tg:
            print(f"decode             : median {statistics.median(tg):.2f} t/s "
                  f"(first {tg[0]:.2f}, last {tg[-1]:.2f})")
        print(f"wall/turn          : median {statistics.median(r['wall'] for r in turns):.1f} s")
        print("\nIf 'reproc' tracks the appended chunk, prefix reuse works. If it "
              "tracks the whole context, the cache is being invalidated and that "
              "dominates every other number in the write-up.")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
