#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Build a GGUF with the DeepSeek-V4 DSpark draft weights (mtp.0/1/2 chained
# blocks + markov head) from a local REAP safetensors checkpoint
# (shards 00046..00048). CPU-only, no downloads.
#
#   python scripts/ds4-convert-mtp.py --dry-run
#   python scripts/ds4-convert-mtp.py
#
# Reuses the proven dequant/quant helpers from convert_hf_to_gguf.DeepseekV4Model.

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LLAMA_CPP_LIBGGML", str(REPO_ROOT / "build-v4-cuda" / "bin" / "libggml.so"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import numpy as np
import torch
from safetensors import safe_open

import gguf
from convert_hf_to_gguf import DeepseekV4Model

N_EXPERTS = 160
BLOCKS = (0, 1, 2)
Q_EXPERTS = getattr(gguf.GGMLQuantizationType, os.environ.get("DSV4_MTP_QEXPERTS", "Q4_K"))
Q_DENSE = gguf.GGMLQuantizationType.Q8_0

DEFAULT_SRC = Path("/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP")
DEFAULT_OUT = Path(os.environ.get(
    "DSV4_MTP_OUT",
    "/mnt/ssd/models/DeepSeek-V4-Flash-0731-REAP-DSPARK-Q4K-Q8_0.gguf"))


def per_block_tables():
    """(gguf_suffix, src_suffix, kind, torch_shape) for each mtp.{b} block."""
    f32 = [
        ("attn_norm.weight",      "attn_norm.weight",      (4096,)),
        ("attn_q_a_norm.weight",  "attn.q_norm.weight",    (1024,)),
        ("attn_kv_a_norm.weight", "attn.kv_norm.weight",   (512,)),
        ("attn_sinks.weight",     "attn.attn_sink",        (64,)),
        ("hc_attn_fn.weight",     "hc_attn_fn",            (24, 16384)),
        ("hc_attn_scale.weight",  "hc_attn_scale",         (3,)),
        ("hc_attn_base.weight",   "hc_attn_base",          (24,)),
        ("hc_ffn_fn.weight",      "hc_ffn_fn",             (24, 16384)),
        ("hc_ffn_scale.weight",   "hc_ffn_scale",          (3,)),
        ("hc_ffn_base.weight",    "hc_ffn_base",           (24,)),
        ("ffn_norm.weight",       "ffn_norm.weight",       (4096,)),
        ("ffn_gate_inp.weight",   "ffn.gate.weight",       (160, 4096)),
        ("exp_probs_b.bias",      "ffn.gate.bias",         (160,)),
    ]
    fp8 = [
        ("attn_q_a.weight",       "attn.wq_a.weight",      (1024, 4096)),
        ("attn_q_b.weight",       "attn.wq_b.weight",      (32768, 1024)),
        ("attn_kv.weight",        "attn.wkv.weight",       (512, 4096)),
        ("attn_output_a.weight",  "attn.wo_a.weight",      (8192, 4096)),
        ("attn_output_b.weight",  "attn.wo_b.weight",      (4096, 8192)),
        ("ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight", (2048, 4096)),
        ("ffn_up_shexp.weight",   "ffn.shared_experts.w3.weight", (2048, 4096)),
        ("ffn_down_shexp.weight", "ffn.shared_experts.w2.weight", (4096, 2048)),
    ]
    return f32, fp8


# block-specific extras: (gguf_name, src_name, kind, torch_shape)
_EXTRAS = [
    ("mtp.0.main_norm.weight",     "mtp.0.main_norm.weight",     "f32", (4096,)),
    ("mtp.0.main_proj.weight",     "mtp.0.main_proj.weight",     "fp8", (4096, 12288)),
    ("mtp.2.hc_head_fn.weight",    "mtp.2.hc_head_fn",           "f32", (4, 16384)),
    ("mtp.2.hc_head_scale.weight", "mtp.2.hc_head_scale",        "f32", (1,)),
    ("mtp.2.hc_head_base.weight",  "mtp.2.hc_head_base",         "f32", (4,)),
    ("mtp.2.norm.weight",          "mtp.2.norm.weight",          "f32", (4096,)),
    ("mtp.2.markov_w1.weight",     "mtp.2.markov_head.markov_w1.weight", "f16", (129280, 256)),
    ("mtp.2.markov_w2.weight",     "mtp.2.markov_head.markov_w2.weight", "f16", (129280, 256)),
]

_EXPERT_WIDS = [("ffn_gate_exps.weight", "w1", (2048, 4096)),
                ("ffn_up_exps.weight",   "w3", (2048, 4096)),
                ("ffn_down_exps.weight", "w2", (4096, 2048))]


class ShardReader:
    """Lazy per-shard safe_open keyed by the checkpoint index."""

    def __init__(self, src_dir: Path):
        with open(src_dir / "model.safetensors.index.json") as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self.src_dir = src_dir
        self.handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"tensor {name!r} not in checkpoint index")
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.src_dir / shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)


def load_block_size(src_dir: Path) -> tuple[int, int]:
    with open(src_dir / "config.json") as f:
        cfg = json.load(f)
    bs = (cfg.get("quantization_config") or {}).get("weight_block_size", [128, 128])
    return int(bs[0]), int(bs[1])


def all_tensors():
    """Yield (gguf_name, src_name, kind, torch_shape) for non-expert tensors."""
    f32, fp8 = per_block_tables()
    for b in BLOCKS:
        for suffix, src_suffix, shape in f32:
            yield f"mtp.{b}.{suffix}", f"mtp.{b}.{src_suffix}", "f32", shape
        for suffix, src_suffix, shape in fp8:
            yield f"mtp.{b}.{suffix}", f"mtp.{b}.{src_suffix}", "fp8", shape
    yield from _EXTRAS


def all_experts():
    """Yield (gguf_name, block, wid, torch_shape)."""
    for b in BLOCKS:
        for suffix, wid, shape in _EXPERT_WIDS:
            yield f"mtp.{b}.{suffix}", b, wid, shape


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


def quantize_np(f32: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    return DeepseekV4Model._quantize_deepseek4_expert(np.ascontiguousarray(f32, dtype=np.float32), qtype)


def convert(reader: ShardReader, out_path: Path, block_size: tuple[int, int], workers: int) -> None:
    writer = gguf.GGUFWriter(out_path, arch="deepseek4", use_temp_file=False)
    writer.add_name("DeepSeek-V4-Flash-0731-REAP-DSPARK")

    for gguf_name, src, kind, shape in all_tensors():
        t = reader.get(src)
        assert tuple(t.shape) == shape, f"{src}: shape {tuple(t.shape)} != expected {shape}"
        if kind == "f32":
            data = to_f32(t)
            print(f"{gguf_name:36s} {t.dtype} -> F32, ggml shape = {tuple(reversed(data.shape))}")
            writer.add_tensor(gguf_name, data)
        elif kind == "f16":
            data = t.detach().to(torch.float16).contiguous().numpy()
            print(f"{gguf_name:36s} {t.dtype} -> F16, ggml shape = {tuple(reversed(data.shape))}")
            writer.add_tensor(gguf_name, data)
        elif kind == "fp8":
            scale = reader.get(src.removesuffix(".weight") + ".scale")
            f32 = DeepseekV4Model._dequant_fp8_weight(t, scale, block_size).numpy()
            data = quantize_np(f32, Q_DENSE)
            qshape = gguf.quant_shape_from_byte_shape(data.shape, Q_DENSE)
            print(f"{gguf_name:36s} FP8 -> {Q_DENSE.name}, ggml shape = {tuple(reversed(qshape))}")
            writer.add_tensor(gguf_name, data, raw_dtype=Q_DENSE)
        else:
            raise ValueError(kind)

    for gguf_name, b, wid, shape in all_experts():
        def convert_one(xid: int) -> tuple[int, np.ndarray]:
            base = f"mtp.{b}.ffn.experts.{xid}.{wid}"
            weight = reader.get(f"{base}.weight")
            scale = reader.get(f"{base}.scale")
            f32 = DeepseekV4Model._dequant_fp4_weight(weight, scale).numpy()
            return xid, quantize_np(f32, Q_EXPERTS)

        experts: dict[int, np.ndarray] = {}
        max_pending = max(1, workers) * 2
        pending: dict[concurrent.futures.Future, int] = {}
        xids = iter(range(N_EXPERTS))
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            def submit_next() -> bool:
                try:
                    xid = next(xids)
                except StopIteration:
                    return False
                pending[executor.submit(convert_one, xid)] = xid
                return True

            while len(pending) < max_pending and submit_next():
                pass
            while pending:
                finished, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    xid = pending.pop(future)
                    experts[xid] = future.result()[1]
                    done += 1
                    if done % 64 == 0 or done == N_EXPERTS:
                        print(f"  {gguf_name}: {done}/{N_EXPERTS} experts", flush=True)
                    submit_next()

        merged = np.stack([experts[i] for i in range(N_EXPERTS)], axis=0)
        del experts
        qshape = gguf.quant_shape_from_byte_shape(merged.shape, Q_EXPERTS)
        assert tuple(qshape) == (N_EXPERTS, *shape), f"{gguf_name}: {tuple(qshape)} != {(N_EXPERTS, *shape)}"
        print(f"{gguf_name:36s} FP4 -> {Q_EXPERTS.name}, ggml shape = {tuple(reversed(qshape))}")
        writer.add_tensor(gguf_name, merged, raw_dtype=Q_EXPERTS)
        del merged

    print("writing GGUF ...", flush=True)
    writer.write_header_to_file(path=out_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {out_path} ({out_path.stat().st_size / (1 << 30):.2f} GiB)")


def verify(out_path: Path) -> None:
    reader = gguf.GGUFReader(str(out_path))
    tensors = {t.name: t for t in reader.tensors}

    expected: dict[str, tuple[gguf.GGMLQuantizationType, tuple[int, ...]]] = {}
    for gguf_name, _, kind, shape in all_tensors():
        q = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
             "fp8": Q_DENSE}[kind]
        expected[gguf_name] = (q, shape)
    for gguf_name, _, _, shape in all_experts():
        expected[gguf_name] = (Q_EXPERTS, (N_EXPERTS, *shape))

    ok = True
    for name, (qtype, torch_shape) in expected.items():
        t = tensors.get(name)
        if t is None:
            print(f"MISSING: {name}")
            ok = False
            continue
        want = tuple(reversed(torch_shape))
        got = tuple(int(d) for d in t.shape)
        if not (got == want and t.tensor_type == qtype):
            print(f"MISMATCH: {name} got {t.tensor_type.name} {got}, want {qtype.name} {want}")
            ok = False

    extra = sorted(set(tensors) - set(expected))
    if extra:
        print(f"unexpected extra tensors: {extra}")
        ok = False

    sinks = tensors["mtp.0.attn_sinks.weight"]
    assert np.isfinite(sinks.data).all(), "attn_sinks contains non-finite values"
    print(f"total tensors: {len(tensors)} (expected {len(expected)})")
    size = out_path.stat().st_size
    print(f"file size: {size / (1 << 30):.2f} GiB (expected ~6.5-7.5 GiB)")
    print("VERIFY:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def dry_run() -> None:
    n = 0
    for gguf_name, src, kind, shape in all_tensors():
        print(f"{gguf_name:36s} {kind:4s} {str(shape):18s} <- {src}")
        n += 1
    for gguf_name, b, wid, shape in all_experts():
        print(f"{gguf_name:36s} q4_k {str((N_EXPERTS, *shape)):18s} <- mtp.{b}.ffn.experts.0..{N_EXPERTS-1}.{wid}")
        n += 1
    print(f"{n} GGUF tensors planned")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="REAP checkpoint dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output GGUF path")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="expert quantization threads")
    ap.add_argument("--dry-run", action="store_true", help="only list/validate tensors, write nothing")
    args = ap.parse_args()

    block_size = load_block_size(args.src)
    print(f"src={args.src} block_size={block_size}")

    if args.dry_run:
        dry_run()
        return

    reader = ShardReader(args.src)
    convert(reader, args.out, block_size, args.workers)
    verify(args.out)


if __name__ == "__main__":
    main()
