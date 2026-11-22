#!/usr/bin/env python3
"""
Streaming mixed-precision 3-bit MoE builder for Qwen3.6-35B-A3B on 24 GB Apple Silicon.

Streams shard-by-shard (< 4 GiB peak build RAM, gc.collect() + mx.clear_cache()
after every shard) from either an unquantized BF16/FP16 checkpoint or a
pre-quantized 4-bit / NVFP4 checkpoint:
- switch_mlp.{gate_proj,up_proj,down_proj} -> 3-bit affine (group_size=64)
- mlp.gate & mlp.shared_expert_gate        -> 8-bit affine (group_size=64)
- attention/GDN/shared_expert/embed/lm_head -> 4-bit affine (group_size=64)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

DEFAULT_M5_MLX_HOME = Path(os.environ.get("M5_MLX_HOME", str(Path.home() / ".local" / "share" / "m5-mlx")))
DEFAULT_DST_DIR = DEFAULT_M5_MLX_HOME / "models" / "Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"

_SKIP_QUANT_SUBSTRINGS = (
    "vision_tower",
    "model.visual",
    "visual.",
    "norm",
    "layernorm",
    "rotary_emb",
    "conv1d",
    "A_log",
    "dt_bias",
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

_TENSOR_SUFFIXES = (
    ".weight",
    ".scales",
    ".biases",
    ".global_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".bias",
)


def _split_module_key(key: str) -> Tuple[str, str]:
    for suffix in _TENSOR_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix[1:]
    return key, ""


def _target_bits_for_module(mod_path: str) -> Optional[int]:
    """Return target affine bit-width (3, 8, or 4) or None if unquantizable."""
    lower = mod_path.lower()
    if any(tok in lower for tok in _SKIP_QUANT_SUBSTRINGS):
        return None
    if mod_path.endswith(
        (
            ".switch_mlp.gate_proj",
            ".switch_mlp.up_proj",
            ".switch_mlp.down_proj",
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.down_proj",
        )
    ):
        return 3
    if mod_path.endswith((".mlp.gate", ".mlp.shared_expert_gate")):
        return 8
    return 4


def _extract_source_quant_info(src_dir: Path, src_cfg: dict) -> Tuple[dict, dict]:
    """Return (global_quant_defaults, per_module_quant_map) from source config."""
    q = src_cfg.get("quantization")
    if not isinstance(q, dict):
        q = src_cfg.get("text_config", {}).get("quantization")
    if isinstance(q, dict):
        defaults = {
            "group_size": int(q.get("group_size", 64)),
            "bits": int(q.get("bits", 4)),
            "mode": str(q.get("mode", "affine")),
        }
        per_mod = {k: v for k, v in q.items() if isinstance(v, dict)}
        return defaults, per_mod

    qcfg = src_cfg.get("quantization_config") or src_cfg.get("text_config", {}).get("quantization_config")
    if not isinstance(qcfg, dict):
        hf_q = src_dir / "hf_quant_config.json"
        if hf_q.exists():
            qcfg = json.loads(hf_q.read_text(encoding="utf-8"))

    if isinstance(qcfg, dict):
        settings = qcfg.get("quantization", qcfg)
        algo = str(settings.get("quant_algo") or qcfg.get("quant_method") or "").upper()
        if "NVFP4" in algo or qcfg.get("quant_method") == "modelopt":
            return {"group_size": int(settings.get("group_size", 16)), "bits": 4, "mode": "nvfp4"}, {}
        return {
            "group_size": int(qcfg.get("group_size", 64)),
            "bits": int(qcfg.get("bits", 4)),
            "mode": str(qcfg.get("mode", "affine")),
        }, {}

    return {"group_size": 64, "bits": 4, "mode": "affine"}, {}


def _lookup_src_module_quant(
    mod_path: str, defaults: dict, per_mod: dict
) -> Tuple[int, int, str]:
    entry = per_mod.get(mod_path)
    if entry is None and mod_path.startswith("language_model."):
        entry = per_mod.get(mod_path[len("language_model.") :])
    if entry is None and mod_path.startswith("model."):
        entry = per_mod.get(mod_path[len("model.") :])
    if isinstance(entry, dict):
        return (
            int(entry.get("bits", defaults["bits"])),
            int(entry.get("group_size", defaults["group_size"])),
            str(entry.get("mode", defaults["mode"])),
        )
    return int(defaults["bits"]), int(defaults["group_size"]), str(defaults["mode"])


def _dequantize_modelopt_nvfp4(
    weight: mx.array,
    weight_scale: mx.array,
    weight_scale_2: Optional[mx.array],
) -> mx.array:
    """Dequantize HF ModelOpt NVFP4 (uint8 packed E2M1 + E4M3 block scales)."""
    if weight.dtype == mx.uint8 and weight.shape[-1] % 4 == 0:
        packed_u32 = weight.view(mx.uint32)
        return mx.dequantize(
            packed_u32,
            weight_scale,
            None,
            group_size=16,
            bits=4,
            mode="nvfp4",
            global_scale=weight_scale_2,
            dtype=mx.float16,
        )

    # Generic fallback for arbitrary uint8 width not divisible by 4 bytes
    lut = mx.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=mx.float16,
    )
    u8 = weight.astype(mx.uint8)
    lo = lut[u8 & 0x0F]
    hi = lut[(u8 >> 4) & 0x0F]
    unpacked = mx.stack([lo, hi], axis=-1).reshape(*weight.shape[:-1], weight.shape[-1] * 2)
    scales_fp = mx.from_fp8(weight_scale, dtype=mx.float16)
    if weight_scale_2 is not None:
        scales_fp = scales_fp * weight_scale_2.astype(mx.float16)
    expanded = mx.repeat(scales_fp, 16, axis=-1)[..., : unpacked.shape[-1]]
    return unpacked * expanded


def _dequantize_to_fp(
    mod_path: str,
    fields: Dict[str, mx.array],
    src_defaults: dict,
    src_per_mod: dict,
) -> mx.array:
    weight = fields["weight"]
    if "scales" in fields:
        s_bits, s_gs, s_mode = _lookup_src_module_quant(mod_path, src_defaults, src_per_mod)
        return mx.dequantize(
            weight,
            fields["scales"],
            fields.get("biases"),
            group_size=s_gs,
            bits=s_bits,
            mode=s_mode,
            global_scale=fields.get("global_scale"),
            dtype=mx.float16,
        )
    if "weight_scale" in fields:
        return _dequantize_modelopt_nvfp4(weight, fields["weight_scale"], fields.get("weight_scale_2"))
    return weight


def _quantize_module(
    mod_path: str,
    fields: Dict[str, mx.array],
    target_bits: Optional[int],
    group_size: int,
    src_defaults: dict,
    src_per_mod: dict,
    quant_entries: Dict[str, Dict[str, object]],
) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}
    if "bias" in fields:
        out[f"{mod_path}.bias"] = fields["bias"]

    if "weight" not in fields:
        for field_name, arr in fields.items():
            if field_name != "bias":
                key = f"{mod_path}.{field_name}" if field_name else mod_path
                out[key] = arr
        return out

    # Handle raw HF fused MoE gate_up_proj if present
    if mod_path.endswith(".mlp.experts.gate_up_proj"):
        fp = _dequantize_to_fp(mod_path, fields, src_defaults, src_per_mod)
        mid = fp.shape[-2] // 2
        base = mod_path[: -len(".mlp.experts.gate_up_proj")] + ".mlp.switch_mlp"
        for sub_name, sub_w in (("gate_proj", fp[..., :mid, :]), ("up_proj", fp[..., mid:, :])):
            sub_mod = f"{base}.{sub_name}"
            if sub_w.ndim >= 2 and sub_w.shape[-1] % group_size == 0:
                q, s, b = mx.quantize(sub_w, group_size=group_size, bits=3, mode="affine")
                out[f"{sub_mod}.weight"] = q
                out[f"{sub_mod}.scales"] = s
                out[f"{sub_mod}.biases"] = b
                quant_entries[sub_mod] = {"group_size": group_size, "bits": 3, "mode": "affine"}
            else:
                out[f"{sub_mod}.weight"] = sub_w
        return out

    if mod_path.endswith(".mlp.experts.down_proj"):
        norm_mod = mod_path[: -len(".mlp.experts.down_proj")] + ".mlp.switch_mlp.down_proj"
        fp = _dequantize_to_fp(mod_path, fields, src_defaults, src_per_mod)
        if fp.ndim >= 2 and fp.shape[-1] % group_size == 0:
            q, s, b = mx.quantize(fp, group_size=group_size, bits=3, mode="affine")
            out[f"{norm_mod}.weight"] = q
            out[f"{norm_mod}.scales"] = s
            out[f"{norm_mod}.biases"] = b
            quant_entries[norm_mod] = {"group_size": group_size, "bits": 3, "mode": "affine"}
        else:
            out[f"{norm_mod}.weight"] = fp
        return out

    # Preserve non-quantizable tensors as-is
    if target_bits is None:
        for field_name, arr in fields.items():
            if field_name != "bias":
                out[f"{mod_path}.{field_name}" if field_name else mod_path] = arr
        return out

    # Fast-path: source module already uses exact target affine quantization
    if "scales" in fields:
        s_bits, s_gs, s_mode = _lookup_src_module_quant(mod_path, src_defaults, src_per_mod)
        if s_bits == target_bits and s_gs == group_size and s_mode == "affine" and "biases" in fields:
            out[f"{mod_path}.weight"] = fields["weight"]
            out[f"{mod_path}.scales"] = fields["scales"]
            out[f"{mod_path}.biases"] = fields["biases"]
            quant_entries[mod_path] = {"group_size": group_size, "bits": target_bits, "mode": "affine"}
            return out

    fp = _dequantize_to_fp(mod_path, fields, src_defaults, src_per_mod)
    if fp.ndim < 2 or fp.shape[-1] % group_size != 0:
        out[f"{mod_path}.weight"] = fp
        return out

    q, s, b = mx.quantize(fp, group_size=group_size, bits=target_bits, mode="affine")
    out[f"{mod_path}.weight"] = q
    out[f"{mod_path}.scales"] = s
    out[f"{mod_path}.biases"] = b
    quant_entries[mod_path] = {"group_size": group_size, "bits": target_bits, "mode": "affine"}
    return out


def build_qwen36_3bit_moe(src_dir: Path, dst_dir: Path, group_size: int = 64) -> dict:
    """Stream-convert a Qwen3.6 MoE checkpoint into 3-bit/8-bit/4-bit mixed-precision MLX format."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported group_size={group_size}; expected 32, 64, or 128")

    dst_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = src_dir / "config.json"
    src_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    src_defaults, src_per_mod = _extract_source_quant_info(src_dir, src_cfg)

    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
        src_weight_map: Dict[str, str] = idx_data["weight_map"]
        shard_names: List[str] = list(dict.fromkeys(src_weight_map.values()))
        total_params = idx_data.get("metadata", {}).get("total_parameters", 0)
    else:
        shard_files = sorted(src_dir.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files found in {src_dir}")
        shard_names = [p.name for p in shard_files]
        src_weight_map = {}
        total_params = 0

    # Track last shard containing each module prefix so cross-shard tensors stay bounded
    mod_last_shard: Dict[str, int] = {}
    if src_weight_map:
        shard_to_idx = {name: idx for idx, name in enumerate(shard_names)}
        for key, sname in src_weight_map.items():
            mod_path, _ = _split_module_key(key)
            sidx = shard_to_idx[sname]
            if mod_path not in mod_last_shard or sidx > mod_last_shard[mod_path]:
                mod_last_shard[mod_path] = sidx

    pending_modules: Dict[str, Dict[str, mx.array]] = {}
    quant_entries: Dict[str, Dict[str, object]] = {}
    out_weight_map: Dict[str, str] = {}
    total_bytes = 0
    bit_counts = {3: 0, 4: 0, 8: 0, 0: 0}

    for shard_idx, shard_name in enumerate(shard_names):
        print(f"[build_qwen36_3bit_moe] Streaming shard {shard_name}...", flush=True)
        shard_tensors = mx.load(str(src_dir / shard_name))
        touched_mods: List[str] = []

        for key, arr in shard_tensors.items():
            mod_path, field = _split_module_key(key)
            if mod_path not in pending_modules:
                pending_modules[mod_path] = {}
                touched_mods.append(mod_path)
            elif mod_path not in touched_mods:
                touched_mods.append(mod_path)
            pending_modules[mod_path][field] = arr

        del shard_tensors

        out_tensors: Dict[str, mx.array] = {}
        for mod_path in touched_mods:
            last_idx = mod_last_shard.get(mod_path, shard_idx)
            if shard_idx < last_idx:
                continue
            fields = pending_modules.pop(mod_path)
            target_bits = _target_bits_for_module(mod_path)
            converted = _quantize_module(
                mod_path=mod_path,
                fields=fields,
                target_bits=target_bits,
                group_size=group_size,
                src_defaults=src_defaults,
                src_per_mod=src_per_mod,
                quant_entries=quant_entries,
            )
            for out_k, out_v in converted.items():
                out_tensors[out_k] = out_v
                out_weight_map[out_k] = shard_name
                total_bytes += out_v.nbytes
            if mod_path in quant_entries:
                b = int(quant_entries[mod_path]["bits"])
                bit_counts[b] = bit_counts.get(b, 0) + 1
            else:
                bit_counts[0] += 1
            del fields, converted

        if out_tensors:
            mx.save_safetensors(str(dst_dir / shard_name), out_tensors)
        del out_tensors
        gc.collect()
        mx.clear_cache()

    # Flush any residual pending modules if weight_map was absent or incomplete
    if pending_modules:
        tail_shard = shard_names[-1]
        tail_tensors: Dict[str, mx.array] = {}
        for mod_path, fields in list(pending_modules.items()):
            target_bits = _target_bits_for_module(mod_path)
            converted = _quantize_module(
                mod_path=mod_path,
                fields=fields,
                target_bits=target_bits,
                group_size=group_size,
                src_defaults=src_defaults,
                src_per_mod=src_per_mod,
                quant_entries=quant_entries,
            )
            for out_k, out_v in converted.items():
                tail_tensors[out_k] = out_v
                out_weight_map[out_k] = tail_shard
                total_bytes += out_v.nbytes
        if tail_tensors:
            existing = mx.load(str(dst_dir / tail_shard)) if (dst_dir / tail_shard).exists() else {}
            existing.update(tail_tensors)
            mx.save_safetensors(str(dst_dir / tail_shard), existing)
            del existing, tail_tensors
        pending_modules.clear()
        gc.collect()
        mx.clear_cache()

    # Build updated config.json
    quant_block: Dict[str, object] = {"group_size": group_size, "bits": 4, "mode": "affine"}
    quant_block.update(quant_entries)
    out_cfg = dict(src_cfg)
    out_cfg["quantization"] = quant_block
    out_cfg["quantization_config"] = {"group_size": group_size, "bits": 4, "mode": "affine"}
    if isinstance(out_cfg.get("text_config"), dict):
        tc = dict(out_cfg["text_config"])
        tc.pop("quantization_config", None)
        tc.pop("quantization", None)
        out_cfg["text_config"] = tc

    (dst_dir / "config.json").write_text(
        json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    idx_out = {
        "metadata": {
            "total_size": total_bytes,
            "total_parameters": total_params,
        },
        "weight_map": out_weight_map,
    }
    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(idx_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for fname in _TOKENIZER_FILES:
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_dir / fname)

    summary = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "group_size": group_size,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "quantized_modules": len(quant_entries),
        "bit_counts": bit_counts,
    }
    print(f"[build_qwen36_3bit_moe] Done: {json.dumps(summary)}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build mixed-precision 3-bit MoE MLX checkpoint for Qwen3.6-35B-A3B."
    )
    parser.add_argument(
        "--src",
        "--source",
        dest="src",
        type=Path,
        default=None,
        help="Local source model directory",
    )
    parser.add_argument(
        "--dst",
        "--output",
        dest="dst",
        type=Path,
        default=DEFAULT_DST_DIR,
        help=f"Destination MLX checkpoint directory (default: {DEFAULT_DST_DIR})",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="",
        help="Optional Hugging Face repo ID to download if --src is not provided",
    )
    parser.add_argument("--group-size", type=int, default=64, help="Affine quantization group size (default: 64)")
    args = parser.parse_args(argv)

    src_dir = args.src
    if src_dir is None:
        if not args.hf_repo:
            parser.error("Provide either --src <local_dir> or --hf-repo <repo_id>")
        from huggingface_hub import snapshot_download

        print(f"[build_qwen36_3bit_moe] Downloading {args.hf_repo}...", flush=True)
        src_dir = Path(snapshot_download(repo_id=args.hf_repo))

    build_qwen36_3bit_moe(src_dir=src_dir, dst_dir=args.dst, group_size=args.group_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
