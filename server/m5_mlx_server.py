#!/usr/bin/env python3
"""
High-Performance OpenAI-Compatible Local LLM Server for Apple M5 MacBook Air (24GB Unified Memory).
Supports three unified-memory-safe backends behind a single `/v1` endpoint:
1. `omlx`: oMLX v0.8.2 (`omlx serve`) with memory guard, SSD paged cache, and serialized sticky
   `llama-server` long-context overflow + idle restore.
2. `llama-cpp`: Supervised `llama-server` with quantized KV cache (`q4_0`), flash attention,
   and optional draft-model speculative decoding.
3. `mlx`: Native `NativeMLXEngine` with compiled Hadamard kernels (`fwht_compiled_fp32` /
   `OptimizedPacked`), two-tier `PrefixCacheManager` (Attention KV + Gated Delta Net state
   checkpoints), bounded Metal wired/scratch limits, keepwarm thread, anti-loop logits processor,
   and XML/JSON tool-call parser with JSON Schema type coercion.
"""

import argparse
import asyncio
import atexit
import concurrent.futures
import copy
import fcntl
import gc
import http.client
import json
import math
import os
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2.sandbox import ImmutableSandboxedEnvironment
from mlx_lm.generate import generate_step, maybe_quantize_kv_cache
from mlx_lm.models.cache import RotatingKVCache  # noqa: F401
from mlx_lm.sample_utils import make_sampler
from tokenizers import Tokenizer


def _get_m5_home() -> Path:
    raw = os.environ.get("M5_MLX_HOME", "").strip()
    return Path(raw) if raw else (Path.home() / ".local" / "share" / "m5-mlx")


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


PACK_DIR = Path(
    os.environ.get(
        "M5_MLX_MODEL_DIR",
        str(_get_m5_home() / "models" / "Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"),
    )
)
if (PACK_DIR / "runtime").exists():
    sys.path.insert(0, str(PACK_DIR / "runtime"))

_SERVER_LOCK_FH = None
_MLX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")
_INKLING_VALID_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_VALID_BACKENDS = frozenset({"mlx", "llama-cpp", "omlx"})


def resolve_inkling_reasoning_effort(
    reasoning_effort: Any = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Map OpenAI reasoning_effort and enable_thinking to Inkling 2 valid efforts."""
    if enable_thinking is False:
        return "none"
    if reasoning_effort is not None:
        val = str(reasoning_effort).strip().lower()
        if val in ("off", "false", "0", "none", "disabled"):
            return "none"
        if val in _INKLING_VALID_EFFORTS:
            return val
    return "low"


def validate_backend_config(
    backend: str = "mlx",
    llama_server_bin: Optional[Path] = None,
    model_file: Optional[Path] = None,
    draft_model_file: Optional[Path] = None,
    llama_port: int = 8081,
    spec_draft_max: int = 0,
    kv_cache_type: str = "q4_0",
    omlx_bin: Optional[Path] = None,
    omlx_model_dir: Optional[Path] = None,
    omlx_port: int = 8081,
    model_id: Optional[str] = None,
    require_paths: bool = False,
) -> Dict[str, Any]:
    """Validate server backend selection and fail closed on invalid configuration."""
    b = str(backend or "").strip().lower()
    if b not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}'; must be one of {sorted(_VALID_BACKENDS)}")
    if not (1 <= int(llama_port) <= 65535):
        raise ValueError(f"llama_port must be 1..65535, got {llama_port}")
    if not (1 <= int(omlx_port) <= 65535):
        raise ValueError(f"omlx_port must be 1..65535, got {omlx_port}")
    if int(spec_draft_max) < 0:
        raise ValueError(f"spec_draft_max must be >= 0, got {spec_draft_max}")
    kv_t = str(kv_cache_type or "q4_0").strip().lower()
    if kv_t not in ("f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"):
        raise ValueError(f"Unsupported kv_cache_type: {kv_cache_type!r}")
    if b == "llama-cpp" and require_paths:
        if not llama_server_bin or not Path(llama_server_bin).is_file():
            raise FileNotFoundError(f"Missing llama-server binary: {llama_server_bin}")
        if not model_file or not Path(model_file).is_file():
            raise FileNotFoundError(f"Missing GGUF model file: {model_file}")
        if draft_model_file and int(spec_draft_max) > 0 and not Path(draft_model_file).is_file():
            raise FileNotFoundError(f"Missing draft GGUF model file: {draft_model_file}")
    if b == "omlx" and require_paths:
        if not omlx_bin or not Path(omlx_bin).is_file():
            raise FileNotFoundError(f"Missing omlx binary: {omlx_bin}")
        if not omlx_model_dir or not Path(omlx_model_dir).is_dir():
            raise FileNotFoundError(f"Missing omlx model dir: {omlx_model_dir}")
        if model_id:
            target_dir = Path(omlx_model_dir) / str(model_id)
            if not target_dir.is_dir():
                raise FileNotFoundError(f"Missing omlx model directory: {target_dir}")
    return {
        "backend": b,
        "llama_port": int(llama_port),
        "omlx_port": int(omlx_port),
        "spec_draft_max": int(spec_draft_max),
        "kv_cache_type": kv_t,
    }


def acquire_single_instance_lock(lock_path: Optional[Path] = None):
    """Acquire an exclusive non-blocking flock guard before allocating Metal RAM."""
    path = Path(lock_path) if lock_path is not None else (_get_m5_home() / "m5_mlx_server.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        fh.close()
        raise RuntimeError(f"Another m5_mlx_server instance already holds {path}") from exc
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def normalize_bash_command_for_loop_check(cmd: str) -> str:
    """Normalize a shell command by stripping echo banners, stderr redirects, and head/tail pipes."""
    s = str(cmd or "").strip()
    s = re.sub(r'(?:^|[;&|]\s*)\becho\s+(?:"[^"]*"|\'[^\']*\'|[^\s;&|]+)\s*', " ", s)
    s = re.sub(r"\b2\s*>\s*(?:/dev/null|&1)\b", " ", s)
    s = re.sub(r"\|\s*(?:head|tail)(?:\s+(?:-n\s*)?\d+)?\b", " ", s)
    s = re.sub(r"\s*[;&]+\s*$", "", s)
    s = re.sub(r"^\s*[;&|]+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _parse_tool_args_for_loop_check(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        parsed = _repair_and_parse_json(raw_args, expected_type="object")
        if isinstance(parsed, dict):
            return parsed
        return {"raw": raw_args}
    return {}


def detect_semantic_tool_loop(messages: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Detect >= 2 near-duplicate tool calls (bash variants or identical MCP calls)
    within the current user turn.
    """
    if not messages:
        return {"is_loop": False, "repeat_count": 0, "tool": None}

    turn_start = 0
    for idx in range(len(messages) - 1, -1, -1):
        m = messages[idx]
        if m.get("role") == "user":
            c_str = str(m.get("content") or "").strip()
            if not (c_str.startswith("<tool_response>") and c_str.endswith("</tool_response>")):
                turn_start = idx + 1
                break

    calls: List[Tuple[str, str, set, str]] = []
    for m in messages[turn_start:]:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            fn_name = str(fn.get("name") or "").strip()
            if not fn_name:
                continue
            args_dict = _parse_tool_args_for_loop_check(fn.get("arguments"))
            if fn_name in ("terminal", "bash", "exec"):
                raw_cmd = str(args_dict.get("command") or args_dict.get("cmd") or "")
                norm_sig = normalize_bash_command_for_loop_check(raw_cmd)
                primary = re.split(r"\s*(?:;|&&|\|\|?)\s*", norm_sig, maxsplit=1)[0].strip()
                tok_set = set(re.findall(r"[a-z0-9_./*~-]+", norm_sig))
            else:
                norm_sig = json.dumps(args_dict, sort_keys=True, ensure_ascii=False).lower()
                primary = norm_sig
                tok_set = set(re.findall(r"[a-z0-9_./*~-]+", f"{fn_name.lower()} {norm_sig}"))
            calls.append((fn_name, norm_sig, tok_set, primary))

    if len(calls) < 2:
        return {"is_loop": False, "repeat_count": len(calls), "tool": calls[-1][0] if calls else None}

    recent = calls[-8:]
    last_fn, last_sig, last_toks, last_primary = recent[-1]
    matches = 0
    for prev_fn, prev_sig, prev_toks, prev_primary in recent[:-1]:
        if prev_fn != last_fn:
            continue
        if last_sig and prev_sig == last_sig:
            matches += 1
            continue
        if last_fn in ("terminal", "bash", "exec") and len(last_primary) >= 8 and prev_primary == last_primary:
            matches += 1
            continue
        union = prev_toks | last_toks
        if union and (len(prev_toks & last_toks) / len(union)) >= 0.65:
            matches += 1

    repeat_count = 1 + matches
    return {
        "is_loop": repeat_count >= 2,
        "repeat_count": repeat_count,
        "tool": last_fn,
        "signature": last_sig,
    }


def resolve_sampling_with_loop_guard(
    messages: List[Dict[str, Any]],
    temperature: float,
    repetition_penalty: float,
) -> Tuple[float, float]:
    """Boost temperature and repetition penalty when duplicate-call loop guard triggers."""
    temp = float(temperature)
    rep = float(repetition_penalty)
    for msg in reversed(messages or []):
        if msg.get("role") == "tool":
            content = str(msg.get("content") or "")
            if "consecutive identical call" in content.lower():
                return max(temp, 0.9), max(rep, 1.12)
            break
    if detect_semantic_tool_loop(messages).get("is_loop"):
        return max(temp, 0.9), max(rep, 1.12)
    return temp, rep


def _encode_ids(tokenizer: Any, text: str) -> List[int]:
    if not text or not hasattr(tokenizer, "encode"):
        return []
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(enc, "ids", enc)
    return [int(x) for x in ids] if ids is not None else []


def build_protected_syntax_token_ids(tokenizer: Any) -> set:
    """
    Collect token IDs for XML/JSON/template structural syntax, punctuation, digits, and
    whitespace so anti-loop penalties never corrupt tool-call formatting.
    """
    protected = set()
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
            for tok_str, tid in vocab.items():
                if any(tag in tok_str for tag in ("tool_call", "think", "im_start", "im_end", "endoftext")):
                    protected.add(int(tid))
                    continue
                norm = tok_str.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
                if norm and all(ord(c) < 128 and not c.isalpha() for c in norm):
                    protected.add(int(tid))
        except Exception:
            pass

    structural_pieces = (
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "<think>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<",
        ">",
        "/",
        '"',
        "{",
        "}",
        "[",
        "]",
        ":",
        ",",
        "=",
        "\n",
        "function",
        "parameter",
    )
    for piece in structural_pieces:
        if hasattr(tokenizer, "token_to_id"):
            try:
                tid = tokenizer.token_to_id(piece)
                if tid is not None:
                    protected.add(int(tid))
            except Exception:
                pass
        try:
            ids = _encode_ids(tokenizer, piece)
            if ids:
                if hasattr(tokenizer, "decode"):
                    decoded = str(tokenizer.decode(ids))
                    if piece in decoded or piece.strip() in decoded.strip():
                        protected.update(ids)
                else:
                    protected.update(ids)
        except Exception:
            pass
    return protected


def has_repeated_ngram(token_ids: Any, n: int = 14) -> bool:
    """Return True iff an exact n-token sequence repeats at least twice in token_ids."""
    if n <= 0 or token_ids is None:
        return False
    seq = [int(t) for t in token_ids]
    if len(seq) < n * 2:
        return False
    seen = set()
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i : i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


_STRAY_INVOKE_TAG_RE = re.compile(r"</?invoke\b[^>]*>", re.IGNORECASE)
_STRAY_PARAM_TAG_RE = re.compile(r"</?parameter(?:=[^>]*)?>", re.IGNORECASE)


def _strip_stray_xml_tags_segment(seg: str) -> str:
    seg = _STRAY_INVOKE_TAG_RE.sub("", seg)
    seg = _STRAY_PARAM_TAG_RE.sub("", seg)
    return seg


def strip_stray_xml_closers(text: str) -> str:
    """
    Strip stray Anthropic/XML <invoke...>, </invoke>, and orphan <parameter...>/</parameter>
    tags outside valid <tool_call>...</tool_call> blocks.
    """
    if not text:
        return ""
    parts: List[str] = []
    last = 0
    for m in TOOL_CALL_BLOCK_RE.finditer(text):
        if m.start() > last:
            parts.append(_strip_stray_xml_tags_segment(text[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()
    if last < len(text):
        parts.append(_strip_stray_xml_tags_segment(text[last:]))
    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_verbatim_assistant_echo(
    candidate_text: str,
    prev_assistant_text: str,
    min_match_chars: int = 60,
) -> bool:
    """
    Detect whether candidate_text echoes >= min_match_chars leading normalized
    non-whitespace characters of prev_assistant_text.
    """
    if min_match_chars <= 0 or not candidate_text or not prev_assistant_text:
        return False
    cand_norm = re.sub(r"\s+", "", candidate_text).lower()
    prev_norm = re.sub(r"\s+", "", prev_assistant_text).lower()
    if len(cand_norm) < min_match_chars or len(prev_norm) < min_match_chars:
        return False
    anchor = prev_norm[:min_match_chars]
    return cand_norm.startswith(anchor) or (anchor in cand_norm[: min_match_chars + 32])


def build_prose_mask(vocab_size: int, protected_ids: set, tokenizer: Any = None) -> mx.array:
    """
    Build a 1D boolean mx.array mask (True = penalizable whole-word alphabetic prose
    with len(decoded.strip()) >= 4, False = protected syntax, non-alphabetic, or short BPE subwords).
    """
    mask_list = [True] * vocab_size
    for pid in protected_ids:
        if 0 <= pid < vocab_size:
            mask_list[pid] = False

    if tokenizer is not None:
        decoded_list: Optional[List[str]] = None
        if hasattr(tokenizer, "decode_batch"):
            try:
                decoded_list = tokenizer.decode_batch([[i] for i in range(vocab_size)])
            except Exception:
                decoded_list = None
        if decoded_list is None and hasattr(tokenizer, "decode"):
            try:
                decoded_list = [tokenizer.decode([i]) for i in range(vocab_size)]
            except Exception:
                decoded_list = None
        if decoded_list is None and hasattr(tokenizer, "get_vocab"):
            try:
                id_to_tok = {int(tid): tok_str for tok_str, tid in tokenizer.get_vocab().items()}
                decoded_list = [
                    id_to_tok.get(i, "").replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
                    for i in range(vocab_size)
                ]
            except Exception:
                decoded_list = None

        if decoded_list is not None:
            for tid in range(min(vocab_size, len(decoded_list))):
                if not mask_list[tid]:
                    continue
                stripped = str(decoded_list[tid] or "").strip()
                if len(stripped) < 4 or not stripped.isalpha():
                    mask_list[tid] = False

    mask = mx.array(mask_list, dtype=mx.bool_)
    mx.eval(mask)
    return mask


def extract_seed_token_ids(
    messages: Optional[List[Dict[str, Any]]],
    tokenizer: Any,
    max_tokens: int = 256,
) -> List[int]:
    """
    Extract up to `max_tokens` token IDs from the last 2 assistant messages in the current
    user turn (raw reasoning_content + content + tool_calls) BEFORE historical thinking is stripped.
    """
    if not messages:
        return []
    turn_start = 0
    for idx in range(len(messages) - 1, -1, -1):
        m = messages[idx]
        if m.get("role") == "user":
            c_str = str(m.get("content") or "").strip()
            if not (c_str.startswith("<tool_response>") and c_str.endswith("</tool_response>")):
                turn_start = idx + 1
                break

    asst_msgs = [m for m in messages[turn_start:] if m.get("role") == "assistant"][-2:]
    if not asst_msgs:
        return []

    parts: List[str] = []
    for m in asst_msgs:
        rc = m.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            parts.append(rc.strip())
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
        for tc in (m.get("tool_calls") or []):
            if isinstance(tc, dict):
                fn = tc.get("function", tc)
                parts.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")

    if not parts:
        return []
    ids = _encode_ids(tokenizer, "\n".join(parts))
    return ids[-max_tokens:] if len(ids) > max_tokens else ids


def make_antiloop_logits_processor(
    seed_token_ids: Optional[List[int]],
    protected_ids: Optional[set],
    presence_penalty: float = 1.2,
    frequency_penalty: float = 0.25,
    repetition_penalty: float = 1.05,
    context_size: int = 384,
    prose_mask: Optional[mx.array] = None,
):
    """
    Vectorized Metal logits processor applying presence, frequency, and repetition penalties
    across prior assistant turn seeds + current generated tokens while leaving protected XML/JSON
    syntax tokens strictly unpenalized (delta == 0.0).
    """
    p_pen = float(presence_penalty)
    f_pen = float(frequency_penalty)
    r_pen = float(repetition_penalty)
    prot = protected_ids or set()
    filtered_seed = [int(t) for t in (seed_token_ids or []) if int(t) >= 0 and int(t) not in prot]
    seed_arr = mx.array(filtered_seed, dtype=mx.int32) if filtered_seed else None
    mask_cache: Dict[int, mx.array] = {}
    if prose_mask is not None:
        mask_cache[int(prose_mask.shape[0])] = prose_mask

    if p_pen == 0.0 and f_pen == 0.0 and r_pen <= 1.0:
        return lambda tokens, logits: logits

    def antiloop_processor(tokens, logits):
        squeeze_out = False
        if logits.ndim == 1:
            logits = logits[None, :]
            squeeze_out = True
        vocab_size = int(logits.shape[-1])

        tok_arr = None
        if tokens is not None:
            if isinstance(tokens, mx.array):
                if tokens.size > 0:
                    tok_arr = tokens[-context_size:].astype(mx.int32)
            elif len(tokens) > 0:
                tok_arr = mx.array([int(t) for t in tokens[-context_size:]], dtype=mx.int32)

        if seed_arr is not None and tok_arr is not None:
            comb = mx.concat([seed_arr, tok_arr])
        elif tok_arr is not None:
            comb = tok_arr
        elif seed_arr is not None:
            comb = seed_arr
        else:
            return logits[0] if squeeze_out else logits

        if int(comb.shape[0]) > context_size:
            comb = comb[-context_size:]
        if int(comb.shape[0]) == 0:
            return logits[0] if squeeze_out else logits

        pmask = mask_cache.get(vocab_size)
        if pmask is None:
            if prose_mask is not None:
                base_len = int(prose_mask.shape[0])
                if vocab_size > base_len:
                    pad = mx.zeros((vocab_size - base_len,), dtype=mx.bool_)
                    pmask = mx.concat([prose_mask, pad])
                else:
                    pmask = prose_mask[:vocab_size]
                mx.eval(pmask)
            else:
                pmask = build_prose_mask(vocab_size, prot)
            mask_cache[vocab_size] = pmask

        is_prose = pmask[comb][None, :]
        dtype = logits.dtype
        is_prose_f = is_prose.astype(dtype)

        if r_pen > 1.0:
            sel = logits[:, comb]
            pen = mx.where(sel < 0, sel * r_pen, sel / r_pen)
            logits[:, comb] = mx.where(is_prose, pen, sel)
        if p_pen != 0.0:
            sel = logits[:, comb]
            logits[:, comb] = sel - (p_pen * is_prose_f)
        if f_pen != 0.0:
            logits = logits.at[:, comb].subtract(f_pen * is_prose_f)

        return logits[0] if squeeze_out else logits

    return antiloop_processor


MODEL_ALIASES = {
    "inkling-2-mini-oq3e-mtp",
    "inkling-2-mini-ud-q3-k-xl",
    "inkling-2-mini",
    "unsloth/Inkling-2-Mini-GGUF",
    "mlx-community/Inkling-2-Mini-oQ3e-mtp",
    "qwen3.6-35b-a3b-3bit-mlx",
    "qwen3.6-35b-a3b-nvfp4-mlx",
    "qwen3.6-35b-a3b-nvfp4",
}


def validate_kv_config(
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    context_length: int = 131072,
    prefill_step_size: int = 256,
) -> Dict[str, Any]:
    """Validate KV cache quantization and context window parameters."""
    if isinstance(kv_bits, str):
        s = kv_bits.strip().lower()
        kv_bits = None if s in ("", "none", "null", "fp16", "bf16", "0") else int(s)
    if kv_bits is not None and kv_bits not in (4, 8):
        raise ValueError(f"kv_bits must be None, 4, or 8 (got {kv_bits!r})")
    kv_group_size = int(kv_group_size)
    if kv_group_size not in (32, 64, 128):
        raise ValueError(f"kv_group_size must be 32, 64, or 128 (got {kv_group_size!r})")
    quantized_kv_start = int(quantized_kv_start)
    if quantized_kv_start < 0:
        raise ValueError(f"quantized_kv_start must be >= 0 (got {quantized_kv_start!r})")
    context_length = int(context_length)
    if context_length < 512 or context_length > 262144:
        raise ValueError(f"context_length must be between 512 and 262144 (got {context_length!r})")
    prefill_step_size = int(prefill_step_size)
    if prefill_step_size < 16 or prefill_step_size > 4096:
        raise ValueError(f"prefill_step_size must be between 16 and 4096 (got {prefill_step_size!r})")
    return {
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
        "context_length": context_length,
        "prefill_step_size": prefill_step_size,
    }


@mx.compile
def fwht_compiled_fp32(x, signs, scale: float, inverse: bool, block: int):
    shape, dtype = x.shape, x.dtype
    x32 = x.astype(mx.float32)
    if not inverse:
        x32 = x32 * signs
    x32 = mx.hadamard_transform(x32.reshape(-1, block), scale=scale).reshape(shape)
    if inverse:
        x32 = x32 * signs
    return x32.astype(dtype)


_ROT_CACHE = {"x_ref": None, "block": 0, "x_rot": None}


class OptimizedPacked(nn.Module):
    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        self.block = block
        self.signs = signs
        self.scale = (1.0 / math.sqrt(block)) if block else 1.0
        self.embedding = embedding
        self.dtype = dtype

    def __call__(self, x):
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=128,
                    bits=2,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return (
                fwht_compiled_fp32(out, self.signs, self.scale, True, self.block)
                if self.block
                else out
            )
        if self.block:
            if _ROT_CACHE["x_ref"] is x and _ROT_CACHE["block"] == self.block:
                x = _ROT_CACHE["x_rot"]
            else:
                x_rot = fwht_compiled_fp32(x, self.signs, self.scale, False, self.block)
                _ROT_CACHE["x_ref"] = x
                _ROT_CACHE["block"] = self.block
                _ROT_CACHE["x_rot"] = x_rot
                x = x_rot
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )


def clone_cache_list(cache_list):
    """Deep-clone hybrid cache (KVCache, QuantizedKVCache, RotatingKVCache, ArraysCache)."""
    cloned = []
    for c in cache_list:
        c_new = copy.copy(c)
        if hasattr(c, "keys") and hasattr(c, "values"):
            if isinstance(c.keys, (tuple, list)):
                c_new.keys = tuple(mx.array(x) for x in c.keys)
                c_new.values = tuple(mx.array(x) for x in c.values) if c.values is not None else None
            else:
                c_new.keys = mx.array(c.keys) if c.keys is not None else None
                c_new.values = mx.array(c.values) if c.values is not None else None
            c_new.offset = c.offset
            if hasattr(c, "_idx"):
                c_new._idx = c._idx
        elif hasattr(c, "cache"):
            c_new.cache = [mx.array(arr) if arr is not None else None for arr in c.cache]
        cloned.append(c_new)
    return cloned


class PrefixCacheManager:
    """Two-tier LRU Checkpoint Cache for Hybrid Attention + Gated Delta Net states.
    Separates pinned system+tools header checkpoints (`sys_entries`) from active
    conversation turn checkpoints (`turn_entries`) so multi-turn tool calls never
    evict the base system+tools prefix and auxiliary tool-less calls never evict
    or duplicate the main conversation cache.
    """

    def __init__(self, max_sys_entries: int = 2, max_turn_entries: int = 3):
        self.max_sys_entries = max_sys_entries
        self.max_turn_entries = max_turn_entries
        self.sys_entries: "OrderedDict[tuple[int, ...], Any]" = OrderedDict()
        self.turn_entries: "OrderedDict[tuple[int, ...], Any]" = OrderedDict()

    def clear_turn_entries(self) -> None:
        self.turn_entries.clear()

    def find_longest_prefix(
        self,
        token_ids: List[int],
        *,
        allow_turn_cache: bool = True,
        consume_turn_entry: bool = False,
    ):
        """Find cached checkpoint that is a strict prefix of token_ids (leaving >= 1 token to step)."""
        best_len = 0
        best_key = None
        best_pool = None
        n = len(token_ids)
        pools = (self.turn_entries, self.sys_entries) if allow_turn_cache else (self.sys_entries,)
        for pool in pools:
            for key in pool.keys():
                klen = len(key)
                if 0 < klen < n and klen > best_len:
                    if tuple(token_ids[:klen]) == key:
                        best_len = klen
                        best_key = key
                        best_pool = pool
        if best_key is not None and best_pool is not None:
            if best_pool is self.turn_entries and consume_turn_entry:
                return best_len, self.turn_entries.pop(best_key)
            best_pool.move_to_end(best_key)
            return best_len, clone_cache_list(best_pool[best_key])
        return 0, None

    def store(
        self,
        token_ids: List[int],
        cache_list,
        is_sys: bool = False,
        *,
        clone: bool = True,
    ):
        if len(token_ids) < 32 or cache_list is None:
            return
        key = tuple(token_ids)
        pool = self.sys_entries if is_sys else self.turn_entries
        max_cap = self.max_sys_entries if is_sys else self.max_turn_entries
        if not is_sys:
            for old_key in list(pool.keys()):
                if len(old_key) <= len(key) and key[: len(old_key)] == old_key:
                    pool.pop(old_key, None)
        pool[key] = clone_cache_list(cache_list) if clone else cache_list
        pool.move_to_end(key)
        while len(pool) > max_cap:
            pool.popitem(last=False)


def _compact_text(text: Any, max_len: int = 220, max_sentences: int = 2) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    first_para = text.strip().split("\n\n", 1)[0]
    joined = " ".join(ln.strip() for ln in first_para.splitlines() if ln.strip())
    if not joined:
        return ""
    if max_sentences >= 1:
        parts = re.split(r"(?<=[.!?])\s+", joined)
        selected: List[str] = []
        for part in parts[:max_sentences]:
            candidate = " ".join(selected + [part]).strip()
            if len(candidate) <= max_len or not selected:
                selected.append(part)
            else:
                break
        if selected:
            joined = " ".join(selected).strip()
    if len(joined) > max_len:
        joined = joined[: max_len - 1].rstrip() + "…"
    return joined


def _compact_schema_node(node: Any, prop_name: str = "") -> Any:
    if not isinstance(node, dict):
        return node
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k in ("additionalProperties", "default", "minimum"):
            continue
        if k == "description":
            short_d = _compact_text(v, max_len=95, max_sentences=1)
            if short_d:
                out[k] = short_d
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _compact_schema_node(pv, pk) for pk, pv in v.items()}
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            out[k] = [_compact_schema_node(x, prop_name) for x in v]
        elif k == "items":
            if isinstance(v, dict):
                if "anyOf" in v and isinstance(v["anyOf"], list) and len(v["anyOf"]) > 2:
                    merged_props: Dict[str, Any] = {}
                    merged_enums: List[str] = []
                    for branch in v["anyOf"]:
                        if isinstance(branch, dict) and isinstance(branch.get("properties"), dict):
                            for bk, bv in branch["properties"].items():
                                if bk == "action" and isinstance(bv, dict) and "enum" in bv:
                                    for ev in bv["enum"]:
                                        if ev not in merged_enums:
                                            merged_enums.append(ev)
                                elif bk not in merged_props:
                                    merged_props[bk] = _compact_schema_node(bv, bk)
                    if merged_enums:
                        merged_props["action"] = {"type": "string", "enum": merged_enums}
                    out[k] = {"type": "object", "properties": merged_props, "required": ["name", "action"]}
                else:
                    out[k] = _compact_schema_node(v, prop_name)
            elif isinstance(v, list):
                out[k] = [_compact_schema_node(x, prop_name) for x in v]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def compact_tools_for_template(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Compact verbose multi-paragraph tool descriptions while preserving
    tool names, descriptions (up to 220 chars), parameter descriptions (up to 95 chars),
    types, enums, and required lists. Sorts tools deterministically by function name so
    clients share identical `<tools>...</tools>` KV cache prefixes.
    """
    if not tools:
        return None
    compacted: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            compacted.append(t)
            continue
        new_fn: Dict[str, Any] = {"name": fn["name"]}
        short_desc = _compact_text(fn.get("description", ""), max_len=220, max_sentences=4)
        if short_desc:
            new_fn["description"] = short_desc
        if "parameters" in fn and isinstance(fn["parameters"], dict):
            new_fn["parameters"] = _compact_schema_node(fn["parameters"])
        compacted.append({"type": t.get("type", "function"), "function": new_fn})
    compacted.sort(key=lambda x: x.get("function", {}).get("name", ""))
    return compacted if compacted else None


def _repair_unescaped_json_quotes(s: str) -> str:
    """Repair unescaped ASCII double quotes inside JSON string literals."""
    out: List[str] = []
    in_string = False
    escape = False
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                nxt = s[j] if j < n else ""
                is_structural_close = False
                if nxt in ("", ":", "}", "]"):
                    is_structural_close = True
                elif nxt == ",":
                    k = j + 1
                    while k < n and s[k] in " \t\r\n":
                        k += 1
                    after_comma = s[k] if k < n else ""
                    if after_comma in ('"', "{", "[", "]", "}", "t", "f", "n", "-") or after_comma.isdigit():
                        is_structural_close = True
                if is_structural_close:
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        if in_string and ch == "\n":
            out.append("\\n")
        elif in_string and ch == "\r":
            out.append("\\r")
        elif in_string and ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _extract_json_objects_fallback(s: str) -> List[Dict[str, Any]]:
    """Fallback extractor for JSON array of objects when outer brackets or inner strings are malformed."""
    objs: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    for idx, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = s[start : idx + 1]
                    for candidate in (chunk, _repair_unescaped_json_quotes(chunk)):
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                objs.append(parsed)
                                break
                        except Exception:
                            pass
                    start = -1
    return objs


def _repair_and_parse_json(raw_str: str, expected_type: Optional[str] = None) -> Any:
    """Parse JSON string with automatic quote repair for unescaped typographic quotes."""
    stripped = raw_str.strip()
    if not stripped:
        return [] if expected_type == "array" else ({} if expected_type == "object" else raw_str)
    for candidate in (stripped, _repair_unescaped_json_quotes(stripped)):
        try:
            parsed = json.loads(candidate)
            if expected_type == "array" and isinstance(parsed, dict):
                return [parsed]
            return parsed
        except Exception:
            pass
    if expected_type == "array" or stripped.startswith("["):
        extracted = _extract_json_objects_fallback(stripped)
        if extracted or expected_type == "array":
            return extracted
    if expected_type == "object":
        extracted = _extract_json_objects_fallback(stripped)
        return extracted[0] if extracted else raw_str
    return raw_str


def _normalize_tool_call_args(fn_name: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool call arguments for common schema edge cases."""
    if not isinstance(args_dict, dict):
        return {}
    if fn_name == "tool_search":
        if "queries" not in args_dict and "query" in args_dict:
            args_dict["queries"] = args_dict.pop("query")
        q_val = args_dict.get("queries")
        if isinstance(q_val, str):
            parsed_q = _repair_and_parse_json(q_val, expected_type="array")
            q_val = parsed_q if isinstance(parsed_q, list) else [q_val]
        elif isinstance(q_val, dict):
            q_val = [q_val]
        if isinstance(q_val, list):
            flat_queries: List[str] = []
            for item in q_val:
                if isinstance(item, str) and item.strip():
                    flat_queries.append(item.strip())
                elif isinstance(item, dict):
                    for dv in item.values():
                        if isinstance(dv, str) and dv.strip():
                            flat_queries.append(dv.strip())
            args_dict["queries"] = flat_queries or ["general"]
        else:
            args_dict["queries"] = ["general"]
    return args_dict


_BINARY_GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]+")


def _compact_tool_message_deterministic(text: str) -> str:
    """Deterministically clean binary replacement garbage and bound oversized tool outputs
    so a single command cannot flood the context window while keeping KV prefix stability.
    """
    if not text:
        return text
    if "\ufffd" in text:
        text = _BINARY_GARBAGE_RE.sub(" ", text)
    if len(text) > 8000:
        text = (
            text[:4500].rstrip()
            + "\n...[tool output truncated for context window efficiency]...\n"
            + text[-2000:].lstrip()
        )
    return text


def _estimate_messages_tokens(
    norm_messages: Optional[List[Dict[str, Any]]],
    compacted_tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Deterministic character-based token estimate (`ceil(chars/3.5)+8*msgs+16*tools`)."""
    msgs = norm_messages or []
    tools_list = compacted_tools or []
    if not msgs and not tools_list:
        return 0
    total_chars = 0
    for m in msgs:
        if not isinstance(m, dict):
            total_chars += len(str(m))
            continue
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False))
        reasoning = m.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            total_chars += len(reasoning)
        tcs = m.get("tool_calls")
        if tcs:
            total_chars += len(json.dumps(tcs, ensure_ascii=False))
    if tools_list:
        total_chars += len(json.dumps(tools_list, ensure_ascii=False))
    return int(math.ceil(total_chars / 3.5)) + (8 * len(msgs)) + (16 * len(tools_list))


def normalize_messages_for_template(
    messages: List[Dict[str, Any]],
    strip_historical_thinking: bool = True,
    inject_loop_breaker: bool = True,
    serialize_tool_args_for_jinja: bool = True,
) -> List[Dict[str, Any]]:
    """
    1. Merge or relocate any non-initial 'system' messages so Jinja chat templates
       requiring leading system messages never fail.
    2. Strip historical `reasoning_content` / inline `<think>...</think>` from completed
       assistant messages so past `<think>` loops never poison GatedDeltaNet recurrent
       states while preserving PrefixCacheManager prefix invariance.
    3. Parse assistant `tool_calls[].function.arguments` from JSON string into a dict (or
       UTF-8 JSON string when `serialize_tool_args_for_jinja=False` for OpenAI wire).
    4. Resolve `role="tool"` `name` from matching `assistant.tool_calls[].id` (`tool_call_id`)
       with safe `"tool"` fallback for orphan tool messages.
    5. When `detect_semantic_tool_loop(messages)` triggers, append a concise loop-breaker
       directive ONLY to the trailing uncached `tool` message (`final_msgs[-1]`).
    """
    if not messages:
        return [{"role": "user", "content": "Hello"}]

    sys_parts = []
    other_msgs = []
    tool_call_id_to_name: Dict[str, str] = {}
    for m in messages:
        msg = copy.deepcopy(m)
        role = msg.get("role", "user")
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        if role == "system":
            if not other_msgs:
                if msg["content"]:
                    sys_parts.append(str(msg["content"]))
            else:
                other_msgs.append(
                    {"role": "user", "content": f"[System Note]\n{msg['content']}"}
                )
            continue

        if role == "tool":
            if isinstance(msg.get("content"), str):
                msg["content"] = _compact_tool_message_deterministic(msg["content"])
            if not str(msg.get("name") or "").strip():
                tc_id = msg.get("tool_call_id")
                resolved_name = tool_call_id_to_name.get(str(tc_id)) if tc_id is not None else None
                msg["name"] = resolved_name or "tool"

        if role == "assistant":
            if strip_historical_thinking:
                msg["reasoning_content"] = ""
                if isinstance(msg.get("content"), str) and "</think>" in msg["content"]:
                    msg["content"] = re.sub(
                        r"<think>.*?</think>\s*", "", msg["content"], flags=re.DOTALL
                    ).lstrip("\n")

            if msg.get("tool_calls"):
                fixed_tcs = []
                for tc in msg["tool_calls"]:
                    tc_copy = copy.deepcopy(tc)
                    fn = tc_copy.get("function", tc_copy)
                    fn_name = fn.get("name", "")
                    tc_id = tc_copy.get("id")
                    if tc_id is not None and fn_name:
                        tool_call_id_to_name[str(tc_id)] = str(fn_name)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        parsed_args = _repair_and_parse_json(args, expected_type="object")
                        args_dict = parsed_args if isinstance(parsed_args, dict) else {"raw": args}
                    elif isinstance(args, dict):
                        args_dict = copy.deepcopy(args)
                    else:
                        args_dict = {}
                    args_dict = _normalize_tool_call_args(fn_name, args_dict)
                    if serialize_tool_args_for_jinja:
                        fn["arguments"] = {
                            k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                            for k, v in args_dict.items()
                        }
                    else:
                        fn["arguments"] = json.dumps(args_dict, ensure_ascii=False)
                    fixed_tcs.append(tc_copy)
                msg["tool_calls"] = fixed_tcs

        other_msgs.append(msg)

    has_user_query = any(
        m.get("role") == "user"
        and not (
            str(m.get("content", "")).strip().startswith("<tool_response>")
            and str(m.get("content", "")).strip().endswith("</tool_response>")
        )
        for m in other_msgs
    )
    if not has_user_query:
        other_msgs.insert(0, {"role": "user", "content": "Continue."})

    final_msgs = []
    if sys_parts:
        final_msgs.append({"role": "system", "content": "\n\n".join(sys_parts)})
    final_msgs.extend(other_msgs)

    if inject_loop_breaker:
        loop_info = detect_semantic_tool_loop(messages)
        if loop_info.get("is_loop") and final_msgs and final_msgs[-1].get("role") == "tool":
            tool_name = loop_info.get("tool") or "tool"
            rep_n = loop_info.get("repeat_count", 2)
            directive = (
                f"\n\n[SYSTEM LOOP-BREAKER: You have repeated a near-identical '{tool_name}' call "
                f"{rep_n} times in this turn without progress. STOP repeating this command or tool. "
                "Either look up official documentation via web_search/web_extract or `--help`, try a "
                "fundamentally different strategy, or summarize directly to the user what is confirmed vs missing.]"
            )
            final_msgs[-1]["content"] = f"{final_msgs[-1].get('content', '')}{directive}"

    return final_msgs


def coerce_param_value(val_str: str, param_schema: Optional[Dict[str, Any]]) -> Any:
    """Coerce XML parameter string into the type declared by the tool's JSON Schema."""
    stripped = val_str.strip()
    if param_schema:
        ptype = param_schema.get("type")
        if isinstance(ptype, list):
            ptype = next((t for t in ptype if t != "null"), ptype[0] if ptype else "string")
        if ptype == "integer":
            try:
                return int(stripped)
            except Exception:
                pass
        elif ptype == "number":
            try:
                return float(stripped)
            except Exception:
                pass
        elif ptype == "boolean":
            if stripped.lower() in ("true", "1", "yes"):
                return True
            if stripped.lower() in ("false", "0", "no"):
                return False
        elif ptype in ("object", "array"):
            return _repair_and_parse_json(stripped, expected_type=ptype)
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        parsed = _repair_and_parse_json(stripped)
        if not isinstance(parsed, str):
            return parsed
    return val_str


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call>\s*(.*)$", re.DOTALL)
FUNCTION_BLOCK_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*(?:</function>|$)", re.DOTALL)
PARAM_BLOCK_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
INKLING_THINK_RE = re.compile(
    r"<\|content_thinking\|>([\s\S]*?)(?=<\|end_message\|>|<\|content_text\|>|<\|message_model\|>|<\|content_invoke_tool_json\|>|<\|content_model_end_sampling\|>|$)"
)
INKLING_TOOL_RE = re.compile(
    r"(?:<\|message_model\|>\s*([^\s<]+)\s*)?<\|content_invoke_tool_json\|>\s*([\s\S]*?)(?=<\|end_message\|>|<\|content_text\|>|<\|content_thinking\|>|<\|message_model\|>|<\|content_model_end_sampling\|>|$)"
)
INKLING_DELIM_RE = re.compile(
    r"<\|(?:content_text|content_thinking|end_message|content_model_end_sampling|message_model|content_invoke_tool_json)\|>"
)
LEAKED_PREAMBLE_RE = re.compile(
    r"^\s*(?:Tazateľ|Uživatel|Dotaz|User)\s+[^\n]+\n+(?:Odpověd[^\n]*\n+|Odpoved[^\n]*\n+)?\s*---\s*\n+",
    re.DOTALL,
)


def _coerce_tool_args_dict(
    fn_name: str,
    raw_args: Dict[str, Any],
    tool_param_schemas: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Coerce string argument values against tool JSON Schema and normalize args."""
    schema_props = tool_param_schemas.get(fn_name, {})
    coerced: Dict[str, Any] = {}
    for k, v in raw_args.items():
        if isinstance(v, str) and k in schema_props:
            coerced[k] = coerce_param_value(v, schema_props[k])
        else:
            coerced[k] = v
    return _normalize_tool_call_args(fn_name, coerced)


def _extract_inkling_tool_call(
    captured_fn_name: Optional[str],
    payload_str: str,
    tool_param_schemas: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    parsed = _repair_and_parse_json(payload_str.strip(), expected_type="object")
    if not isinstance(parsed, dict):
        return None
    fn_name = parsed.get("name") or (parsed.get("function") or {}).get("name") or captured_fn_name
    if not fn_name:
        return None
    fn_name = str(fn_name).strip()
    if "args" in parsed:
        raw_args = parsed["args"]
    elif "arguments" in parsed:
        raw_args = parsed["arguments"]
    elif "parameters" in parsed:
        raw_args = parsed["parameters"]
    elif captured_fn_name and "name" not in parsed:
        raw_args = parsed
    else:
        raw_args = {}
    if isinstance(raw_args, str):
        raw_args = _repair_and_parse_json(raw_args, expected_type="object")
    if isinstance(raw_args, dict):
        raw_args = _coerce_tool_args_dict(fn_name, raw_args, tool_param_schemas)
        serialized = json.dumps(raw_args, ensure_ascii=False)
    else:
        serialized = json.dumps({}, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": fn_name, "arguments": serialized},
    }


def _extract_single_tool_call(
    inner: str, tool_param_schemas: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    fn_match = FUNCTION_BLOCK_RE.search(inner)
    if fn_match:
        fn_name = fn_match.group(1).strip().strip("\"'")
        fn_body = fn_match.group(2)
        args_dict = {}
        schema_props = tool_param_schemas.get(fn_name, {})
        for p_match in PARAM_BLOCK_RE.finditer(fn_body):
            p_name = p_match.group(1).strip().strip("\"'")
            p_val_raw = p_match.group(2)
            args_dict[p_name] = coerce_param_value(p_val_raw, schema_props.get(p_name))
        args_dict = _normalize_tool_call_args(fn_name, args_dict)
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps(args_dict, ensure_ascii=False),
            },
        }
    try:
        parsed = _repair_and_parse_json(inner, expected_type="object")
        if isinstance(parsed, dict):
            fn_name = parsed.get("name") or parsed.get("function", {}).get("name")
            fn_args = parsed.get("arguments", parsed.get("parameters", {}))
            if isinstance(fn_args, str):
                fn_args = _repair_and_parse_json(fn_args, expected_type="object")
            if isinstance(fn_args, dict) and fn_name:
                fn_args = _normalize_tool_call_args(fn_name, fn_args)
            if not isinstance(fn_args, str):
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            if fn_name:
                return {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }
    except Exception:
        pass
    return None


def parse_model_output(raw_text: str, tools: Optional[List[Dict[str, Any]]] = None):
    """
    Extract `reasoning_content`, clean `content`, and structured `tool_calls` from
    Qwen3.5/Qwen3.6 XML/JSON and Inkling 2 delimiter outputs.
    """
    tool_param_schemas: Dict[str, Dict[str, Any]] = {}
    if tools:
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name")
            props = fn.get("parameters", {}).get("properties", {})
            if name and isinstance(props, dict):
                tool_param_schemas[name] = props

    reasoning_parts = []
    text = raw_text.replace("<|content_model_end_sampling|>", "")
    tool_calls = []

    if (
        "<|content_thinking|>" in text
        or "<|content_text|>" in text
        or "<|content_invoke_tool_json|>" in text
        or "<|message_model|>" in text
        or "<|end_message|>" in text
    ):
        for m in INKLING_THINK_RE.finditer(text):
            if m.group(1).strip():
                reasoning_parts.append(m.group(1).strip())
        text = INKLING_THINK_RE.sub("", text)

        for tm in INKLING_TOOL_RE.finditer(text):
            tc_obj = _extract_inkling_tool_call(tm.group(1), tm.group(2), tool_param_schemas)
            if tc_obj:
                tool_calls.append(tc_obj)
        text = INKLING_TOOL_RE.sub("", text)
        text = INKLING_DELIM_RE.sub("", text)

    if "</think>" in text and "<think>" not in text.split("</think>", 1)[0]:
        first_think, rest = text.split("</think>", 1)
        if first_think.strip():
            reasoning_parts.append(first_think.strip())
        text = rest

    for m in THINK_BLOCK_RE.finditer(text):
        if m.group(1).strip():
            reasoning_parts.append(m.group(1).strip())
    text = THINK_BLOCK_RE.sub("", text)

    if "<think>" in text and "</think>" not in text:
        before_think, unclosed_think = text.split("<think>", 1)
        if unclosed_think.strip():
            reasoning_parts.append(unclosed_think.strip())
        text = before_think

    for tc_match in TOOL_CALL_BLOCK_RE.finditer(text):
        tc_obj = _extract_single_tool_call(tc_match.group(1).strip(), tool_param_schemas)
        if tc_obj:
            tool_calls.append(tc_obj)

    clean_content = TOOL_CALL_BLOCK_RE.sub("", text)
    if "<tool_call>" in clean_content:
        m_unclosed = TOOL_CALL_UNCLOSED_RE.search(clean_content)
        if m_unclosed:
            tc_obj = _extract_single_tool_call(m_unclosed.group(1).strip(), tool_param_schemas)
            if tc_obj:
                tool_calls.append(tc_obj)
            clean_content = clean_content[: m_unclosed.start()]

    clean_content = strip_stray_xml_closers(clean_content)
    if clean_content:
        m_pre = LEAKED_PREAMBLE_RE.match(clean_content)
        if m_pre:
            reasoning_parts.append(m_pre.group(0).strip())
            clean_content = clean_content[m_pre.end() :].strip()
    reasoning_content = "\n\n".join(reasoning_parts) if reasoning_parts else None
    return clean_content, reasoning_content, tool_calls


def compute_wired_limit_bytes(wired_limit_mb: float, rec_max_bytes: int) -> int:
    """Compute bounded Metal wired memory limit clamped to 50% of max recommended working set."""
    if wired_limit_mb <= 0 or rec_max_bytes <= 0:
        return 0
    return min(int(wired_limit_mb * 1024 * 1024), int(0.5 * rec_max_bytes))


def collect_keepwarm_tensors(model: Any) -> List[Any]:
    """Collect tiny weight slices across all MoE layers so macOS does not page out middle layers."""
    layers = getattr(model, "layers", None)
    if not layers:
        return []
    tensors: List[Any] = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for attr in ("gate", "shared_expert_gate"):
            sub = getattr(mlp, attr, None)
            w = getattr(sub, "weight", None) if sub is not None else None
            if w is not None:
                tensors.append(w[:1])
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(switch_mlp, proj_name, None)
                w = getattr(proj, "weight", None) if proj is not None else None
                if w is not None:
                    tensors.append(w[:1])
                    break
    return tensors


class NativeMLXEngine:
    """In-process Apple Silicon MLX inference engine with PrefixCacheManager and wired memory cap."""

    backend = "mlx"
    speculative: Optional[str] = None

    def __init__(
        self,
        pack_dir: Path,
        *,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        context_length: int = 131072,
        prefill_step_size: int = 256,
        model_id: Optional[str] = None,
    ):
        self.pack_dir = Path(pack_dir)
        self.lock = threading.RLock()
        self.prefix_cache = PrefixCacheManager(max_sys_entries=2, max_turn_entries=3)
        self._sys_len_cache: Dict[str, int] = {}
        self._stop_event = threading.Event()

        env_kv_bits = os.environ.get("M5_MLX_KV_BITS")
        if kv_bits is None:
            kv_bits = env_kv_bits if env_kv_bits is not None else 4  # type: ignore[assignment]
        kv_cfg = validate_kv_config(
            kv_bits=kv_bits,
            kv_group_size=int(os.environ.get("M5_MLX_KV_GROUP_SIZE", str(kv_group_size))),
            quantized_kv_start=int(os.environ.get("M5_MLX_QUANTIZED_KV_START", str(quantized_kv_start))),
            context_length=int(os.environ.get("M5_MLX_CONTEXT_LENGTH", str(context_length))),
            prefill_step_size=int(os.environ.get("M5_MLX_PREFILL_STEP_SIZE", str(prefill_step_size))),
        )
        self.kv_bits: Optional[int] = kv_cfg["kv_bits"]
        self.kv_group_size: int = kv_cfg["kv_group_size"]
        self.quantized_kv_start: int = kv_cfg["quantized_kv_start"]
        self.context_length: int = kv_cfg["context_length"]
        self.prefill_step_size: int = kv_cfg["prefill_step_size"]
        self.model_id: str = model_id or os.environ.get("M5_MLX_MODEL_ID", "qwen3.6-35b-a3b-3bit-mlx")

        # Keep MLX scratch buffer pool bounded at 128 MiB so 24 GB Unified Memory has
        # maximum headroom for macOS WindowServer and desktop apps.
        mx.set_cache_limit(128 * 1024**2)

        wired_mb_raw = float(os.environ.get("M5_MLX_WIRED_LIMIT_MB", "6144"))
        rec_max = 0
        try:
            if hasattr(mx, "device_info"):
                rec_max = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
            if rec_max <= 0 and hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
                rec_max = int(mx.metal.device_info().get("max_recommended_working_set_size", 0) or 0)
        except Exception:
            rec_max = 0
        if rec_max <= 0:
            rec_max = 16 * 1024**3
        wired_bytes = compute_wired_limit_bytes(wired_mb_raw, rec_max)
        self.wired_limit_mb: int = int(wired_bytes // (1024 * 1024)) if wired_bytes > 0 else 0
        if wired_bytes > 0 and hasattr(mx, "set_wired_limit"):
            try:
                mx.set_wired_limit(wired_bytes)
            except Exception as exc:
                print(f"[NativeMLXEngine] Warning: mx.set_wired_limit failed: {exc}", flush=True)

        t0 = time.perf_counter()
        self.config = json.loads((self.pack_dir / "config.json").read_text())
        gen_cfg_path = self.pack_dir / "generation_config.json"
        self.gen_config = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
        self.arch = self.config.get("model_type", "qwen3_5")

        if self.arch == "prism_hadamard_qwen35":
            print(
                f"[NativeMLXEngine] Loading Text-Only Compiled Hadamard 2-bit model from {self.pack_dir}...",
                flush=True,
            )
            from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

            self.model = TextModel(TextModelArgs.from_dict(self.config["text_config"]))
            raw_weights = mx.load(str(self.pack_dir / "model.safetensors"))

            for record in self.config["modules"]:
                path = record["path"]
                parts = path.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                key = "language_model." + path
                arrays = [raw_weights[key + "." + s] for s in ("weight", "scales", "biases")]
                block = record["block"]
                signs = raw_weights.get(key + ".signs")
                setattr(
                    parent,
                    parts[-1],
                    OptimizedPacked(arrays, block, signs, record["embedding"], mx.float16),
                )

            lm_weights = [
                (k[len("language_model.") :], v)
                for k, v in raw_weights.items()
                if k.startswith("language_model.")
            ]
            self.model.load_weights(lm_weights, strict=True)
            del raw_weights, lm_weights
            gc.collect()
            self.model.eval()
            mx.eval(self.model.parameters())
        else:
            print(
                f"[NativeMLXEngine] Loading MLX model ({self.arch}) from {self.pack_dir}...",
                flush=True,
            )
            from mlx_lm.utils import load_model

            loaded_model, _ = load_model(self.pack_dir, lazy=False, strict=False)
            self.model = getattr(loaded_model, "language_model", loaded_model)
            gc.collect()

        self.tokenizer = Tokenizer.from_file(str(self.pack_dir / "tokenizer.json"))
        self.jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        ct_path = self.pack_dir / "chat_template.jinja"
        if ct_path.exists():
            ct_source = ct_path.read_text()
        else:
            tc_path = self.pack_dir / "tokenizer_config.json"
            tc_data = json.loads(tc_path.read_text()) if tc_path.exists() else {}
            ct_source = tc_data.get("chat_template", "")
            if not ct_source:
                fallback_ct = (
                    _get_m5_home()
                    / "models"
                    / "Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"
                    / "chat_template.jinja"
                )
                if fallback_ct.exists():
                    ct_source = fallback_ct.read_text()
                else:
                    raise FileNotFoundError(f"No chat_template found in {self.pack_dir}")
        self.template = self.jinja_env.from_string(ct_source)

        eos = self.gen_config.get("eos_token_id", self.config.get("eos_token_id", []))
        self.stop_tokens = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
        for tok_str in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.token_to_id(tok_str)
            if tid is not None:
                self.stop_tokens.add(tid)

        self.protected_syntax_ids = build_protected_syntax_token_ids(self.tokenizer)
        vocab_sz = (
            self.tokenizer.get_vocab_size()
            if hasattr(self.tokenizer, "get_vocab_size")
            else len(self.tokenizer.get_vocab())
        )
        self._prose_mask = build_prose_mask(vocab_sz, self.protected_syntax_ids, tokenizer=self.tokenizer)

        warm_cache = self.model.make_cache()
        mx.eval(self.model(mx.array([[248044, 198]]), cache=warm_cache))
        del warm_cache
        mx.clear_cache()

        dt = time.perf_counter() - t0
        active_gb = mx.get_active_memory() / (1024**3)
        print(
            f"[NativeMLXEngine] Ready in {dt:.2f}s | Model: {self.model_id} ({self.arch}) "
            f"| KV: bits={self.kv_bits}, group={self.kv_group_size}, ctx={self.context_length} "
            f"| Active Metal RAM: {active_gb:.2f} GB",
            flush=True,
        )
        self._start_keepwarm_thread()

    def close(self) -> None:
        self._stop_event.set()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def shutdown(self) -> None:
        self.close()

    def _start_keepwarm_thread(self) -> None:
        """Periodically touch boundary router slices when idle so macOS less aggressively swaps model pages."""
        if os.environ.get("M5_MLX_DISABLE_KEEPWARM") == "1":
            return

        def _do_keepwarm():
            if not self.lock.acquire(blocking=False):
                return
            try:
                tensors = collect_keepwarm_tensors(self.model)
                if tensors:
                    mx.eval(*tensors)
                else:
                    warm_c = self.model.make_cache()
                    mx.eval(self.model(mx.array([[248044]]), cache=warm_c))
                    del warm_c
                mx.clear_cache()
            except Exception:
                pass
            finally:
                self.lock.release()

        def _keepwarm_loop():
            interval_s = max(5, int(os.environ.get("M5_MLX_KEEPWARM_INTERVAL_S", "60")))
            while not self._stop_event.wait(interval_s):
                if self.lock.locked():
                    continue
                try:
                    _MLX_EXECUTOR.submit(_do_keepwarm).result(timeout=10)
                except Exception:
                    pass

        t = threading.Thread(target=_keepwarm_loop, daemon=True, name="m5-mlx-keepwarm")
        t.start()

    def render_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "medium",
        inject_loop_breaker: bool = True,
    ) -> str:
        norm_msgs = normalize_messages_for_template(
            messages, inject_loop_breaker=inject_loop_breaker
        )
        compacted_tools = compact_tools_for_template(tools)
        if reasoning_effort not in ("xhigh", "medium", "low"):
            reasoning_effort = "medium"
        return self.template.render(
            messages=norm_msgs,
            tools=compacted_tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=False,
        )

    @staticmethod
    def _trailing_tag_prefix_len(text: str, tags: Tuple[str, ...]) -> int:
        max_hold = 0
        for tag in tags:
            limit = min(len(text), len(tag) - 1)
            for k in range(limit, 0, -1):
                if text.endswith(tag[:k]) and k > max_hold:
                    max_hold = k
                    break
        return max_hold

    def _prepare_prefill_cache(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        enable_thinking: bool,
        reasoning_effort: str,
        cancel_event: Optional[threading.Event] = None,
    ):
        is_aux = not bool(tools)
        compacted_tools = compact_tools_for_template(tools)
        prompt_str = self.render_prompt(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        if is_aux and "You are a summarization agent creating a context checkpoint" in prompt_str:
            max_tokens = min(max_tokens, 640)

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False).ids

        clean_turn_limit = len(prompt_ids) - 1
        if not is_aux and detect_semantic_tool_loop(messages).get("is_loop"):
            clean_prompt_str = self.render_prompt(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                inject_loop_breaker=False,
            )
            clean_prompt_ids = self.tokenizer.encode(
                clean_prompt_str, add_special_tokens=False
            ).ids
            common_len = 0
            for a, b in zip(clean_prompt_ids, prompt_ids):
                if a != b:
                    break
                common_len += 1
            clean_turn_limit = min(clean_turn_limit, common_len)

        sys_prefix_len = 0
        norm_msgs = normalize_messages_for_template(messages, inject_loop_breaker=False)
        if not is_aux and norm_msgs and norm_msgs[0].get("role") == "system":
            sys_only_str = self.template.render(
                messages=[norm_msgs[0], {"role": "user", "content": "X"}],
                tools=compacted_tools,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            cached_len = self._sys_len_cache.get(sys_only_str)
            if cached_len is not None:
                sys_prefix_len = cached_len
            else:
                marker = "<|im_start|>user\nX<|im_end|>"
                if marker in sys_only_str:
                    sys_header_str = sys_only_str.split(marker, 1)[0] + "<|im_start|>user\n"
                    sys_prefix_len = len(
                        self.tokenizer.encode(sys_header_str, add_special_tokens=False).ids
                    )
                if len(self._sys_len_cache) >= 16:
                    self._sys_len_cache.clear()
                self._sys_len_cache[sys_only_str] = sys_prefix_len

        if len(prompt_ids) > self.context_length - 64:
            prompt_ids = prompt_ids[-(self.context_length - 64) :]
            sys_prefix_len = 0
            clean_turn_limit = len(prompt_ids) - 1

        matched_len, cache = self.prefix_cache.find_longest_prefix(
            prompt_ids,
            allow_turn_cache=not is_aux,
            consume_turn_entry=False,
        )

        if cache is None:
            cache = self.model.make_cache()
            matched_len = 0
        initial_matched_len = matched_len

        def _step_prefill_to(target_len: int, *, is_sys: bool, store_cache: bool = True) -> bool:
            nonlocal matched_len
            prefill_backbone = getattr(self.model, "model", None)
            cursor = matched_len
            while cursor < target_len:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                step_end = min(cursor + self.prefill_step_size, target_len)
                sub = mx.array([prompt_ids[cursor:step_end]])
                if prefill_backbone is not None:
                    hidden = prefill_backbone(sub, cache=cache)
                else:
                    hidden = self.model(sub, cache=cache)
                if self.kv_bits is not None:
                    maybe_quantize_kv_cache(
                        cache,
                        quantized_kv_start=self.quantized_kv_start,
                        kv_group_size=self.kv_group_size,
                        kv_bits=self.kv_bits,
                    )
                mx.eval([c.state for c in cache])
                del hidden, sub
                cursor = step_end
                if cancel_event is not None and cancel_event.is_set():
                    return False
                time.sleep(0.005)
            if store_cache:
                self.prefix_cache.store(prompt_ids[:target_len], cache, is_sys=is_sys, clone=True)
            matched_len = target_len
            return True

        prefill_aborted = bool(cancel_event is not None and cancel_event.is_set())
        if not prefill_aborted and not is_aux:
            if 64 <= sys_prefix_len < len(prompt_ids) and matched_len < sys_prefix_len:
                if not _step_prefill_to(sys_prefix_len, is_sys=True, store_cache=True):
                    prefill_aborted = True

        turn_target_len = min(len(prompt_ids) - 1, clean_turn_limit)
        if not prefill_aborted and turn_target_len >= 64 and matched_len < turn_target_len:
            if not _step_prefill_to(turn_target_len, is_sys=False, store_cache=not is_aux):
                prefill_aborted = True

        remaining_ids = [] if prefill_aborted else prompt_ids[matched_len:]
        return is_aux, max_tokens, prompt_ids, remaining_ids, initial_matched_len, cache

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "medium",
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        with self.lock:
            t0 = time.perf_counter()
            if temperature is None:
                temperature = 0.6
            if top_p is None:
                top_p = 0.95
            if presence_penalty is None:
                presence_penalty = 0.0
            if frequency_penalty is None:
                frequency_penalty = 0.0

            loop_info = detect_semantic_tool_loop(messages)
            temperature, repetition_penalty = resolve_sampling_with_loop_guard(
                messages, temperature, repetition_penalty
            )
            if loop_info.get("is_loop"):
                temperature = max(float(temperature), 0.9)
                presence_penalty = max(float(presence_penalty), 1.5)
                frequency_penalty = max(float(frequency_penalty), 0.4)
                repetition_penalty = max(float(repetition_penalty), 1.08)

            prev_assistant_text = ""
            for m in reversed(messages or []):
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str) and len(re.sub(r"\s+", "", c)) >= 60:
                        prev_assistant_text = c
                    break

            seed_token_ids = extract_seed_token_ids(messages, self.tokenizer, max_tokens=256)

            is_aux, max_tokens, prompt_ids, remaining_ids, initial_matched_len, cache = (
                self._prepare_prefill_cache(
                    messages,
                    tools,
                    max_tokens,
                    enable_thinking,
                    reasoning_effort,
                    cancel_event=cancel_event,
                )
            )

            pass_sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            pass_logits_proc = [
                make_antiloop_logits_processor(
                    seed_token_ids=seed_token_ids,
                    protected_ids=getattr(self, "protected_syntax_ids", None),
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    context_size=384,
                    prose_mask=getattr(self, "_prose_mask", None),
                )
            ]

            max_thinking_tokens: Optional[int] = None
            if enable_thinking:
                if reasoning_effort == "low":
                    max_thinking_tokens = 1024
                else:
                    max_thinking_tokens = 1536

            generated_ids: List[int] = []
            t_first: Optional[float] = None
            mode = "THINK" if enable_thinking else "CONTENT"
            think_start_idx = 0
            strip_leading_nl = False
            cursor = 0
            emitted_content = False
            emitted_reasoning = False
            cancelled = False
            hit_stop = False
            thinking_capped = False
            phase1_len: Optional[int] = None
            phase1_text: Optional[str] = None
            content_echo_buf: List[str] = []
            content_echo_checked = not bool(prev_assistant_text)
            echo_detected = False

            def _emit_or_buffer_content(chunk_text: str) -> Iterator[Dict[str, Any]]:
                nonlocal emitted_content, content_echo_checked, echo_detected
                if echo_detected:
                    return
                cleaned_chunk = _strip_stray_xml_tags_segment(chunk_text)
                if not cleaned_chunk:
                    return
                if content_echo_checked:
                    if cleaned_chunk.strip():
                        emitted_content = True
                    yield {"type": "content", "text": cleaned_chunk}
                    return
                content_echo_buf.append(cleaned_chunk)
                joined = "".join(content_echo_buf)
                non_ws_len = len(re.sub(r"\s+", "", joined))
                if len(joined) >= 72 and non_ws_len >= 60:
                    if is_verbatim_assistant_echo(joined, prev_assistant_text, min_match_chars=60):
                        echo_detected = True
                        return
                    content_echo_checked = True
                    content_echo_buf.clear()
                    if joined.strip():
                        emitted_content = True
                    yield {"type": "content", "text": joined}

            def _drain_safe_text(text_buf: str) -> Iterator[Dict[str, Any]]:
                nonlocal cursor, mode, think_start_idx, strip_leading_nl, emitted_content, emitted_reasoning
                safe_len = len(text_buf)
                while safe_len > cursor and text_buf[safe_len - 1] == "\ufffd":
                    safe_len -= 1

                while cursor < safe_len and not echo_detected:
                    if mode == "TOOL_CALL":
                        break
                    if mode == "THINK":
                        if cursor == 0:
                            rem = text_buf[:safe_len]
                            if rem.startswith("<think>"):
                                cursor = len("<think>")
                                continue
                            if "<think>".startswith(rem):
                                break
                        end_idx = text_buf.find("</think>", cursor, safe_len)
                        if end_idx != -1:
                            r_chunk = text_buf[cursor:end_idx]
                            if r_chunk:
                                emitted_reasoning = True
                                yield {"type": "reasoning", "text": r_chunk}
                            cursor = end_idx + len("</think>")
                            mode = "CONTENT"
                            strip_leading_nl = True
                            continue
                        hold = self._trailing_tag_prefix_len(
                            text_buf[cursor:safe_len], ("</think>",)
                        )
                        emit_end = safe_len - hold
                        if emit_end > cursor:
                            r_chunk = text_buf[cursor:emit_end]
                            emitted_reasoning = True
                            yield {"type": "reasoning", "text": r_chunk}
                            cursor = emit_end
                        break

                    if strip_leading_nl:
                        while cursor < safe_len and text_buf[cursor] == "\n":
                            cursor += 1
                        if cursor < safe_len:
                            strip_leading_nl = False
                        else:
                            break

                    tc_idx = text_buf.find("<tool_call>", cursor, safe_len)
                    th_idx = text_buf.find("<think>", cursor, safe_len)
                    if tc_idx != -1 and (th_idx == -1 or tc_idx < th_idx):
                        c_chunk = text_buf[cursor:tc_idx]
                        if c_chunk:
                            yield from _emit_or_buffer_content(c_chunk)
                        cursor = tc_idx
                        mode = "TOOL_CALL"
                        break
                    if th_idx != -1:
                        c_chunk = text_buf[cursor:th_idx]
                        if c_chunk:
                            yield from _emit_or_buffer_content(c_chunk)
                        cursor = th_idx + len("<think>")
                        mode = "THINK"
                        think_start_idx = len(generated_ids)
                        continue

                    hold = self._trailing_tag_prefix_len(
                        text_buf[cursor:safe_len],
                        ("<tool_call>", "<think>", "<invoke", "</invoke>", "<parameter", "</parameter>"),
                    )
                    emit_end = safe_len - hold
                    if emit_end > cursor:
                        sub_seg = text_buf[cursor:emit_end]
                        last_lt = sub_seg.rfind("<")
                        if last_lt != -1 and ">" not in sub_seg[last_lt:]:
                            tail_tag = sub_seg[last_lt:].lower()
                            if (
                                tail_tag.startswith(("<invoke", "</invoke", "<parameter", "</parameter"))
                                and (len(sub_seg) - last_lt) < 200
                            ):
                                emit_end = cursor + last_lt
                    if emit_end > cursor:
                        c_chunk = text_buf[cursor:emit_end]
                        yield from _emit_or_buffer_content(c_chunk)
                        cursor = emit_end
                    break

            raw_text = ""
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
            try:
                if not cancelled and remaining_ids:
                    for tok, _ in generate_step(
                        mx.array(remaining_ids),
                        self.model,
                        max_tokens=max_tokens,
                        sampler=pass_sampler,
                        logits_processors=pass_logits_proc,
                        prompt_cache=cache,
                        prefill_step_size=self.prefill_step_size,
                        kv_bits=self.kv_bits,
                        kv_group_size=self.kv_group_size,
                        quantized_kv_start=self.quantized_kv_start,
                    ):
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled = True
                            break
                        if t_first is None:
                            t_first = time.perf_counter()
                        tid = int(tok)
                        if tid in self.stop_tokens:
                            hit_stop = True
                            break
                        generated_ids.append(tid)

                        if mode != "TOOL_CALL":
                            raw_text = self.tokenizer.decode(generated_ids)
                            yield from _drain_safe_text(raw_text)
                            if echo_detected:
                                break

                        if mode == "THINK":
                            think_ids = generated_ids[think_start_idx:]
                            if has_repeated_ngram(think_ids, n=14) or (
                                max_thinking_tokens is not None and len(think_ids) >= max_thinking_tokens
                            ):
                                thinking_capped = True
                                break

                if thinking_capped and not cancelled and not hit_stop and not echo_detected:
                    if cursor < len(raw_text):
                        tail_think = raw_text[cursor:]
                        if tail_think:
                            emitted_reasoning = True
                            yield {"type": "reasoning", "text": tail_think}
                    mx.synchronize()
                    phase1_len = len(generated_ids)
                    forced_close = "\n</think>\n\n"
                    phase1_text = self.tokenizer.decode(generated_ids) + forced_close
                    cursor = len(phase1_text)
                    mode = "CONTENT"
                    strip_leading_nl = True
                    remaining_budget = max_tokens - len(generated_ids)
                    if remaining_budget > 0:
                        close_ids = self.tokenizer.encode(
                            forced_close, add_special_tokens=False
                        ).ids
                        for tok, _ in generate_step(
                            mx.array(close_ids),
                            self.model,
                            max_tokens=remaining_budget,
                            sampler=pass_sampler,
                            logits_processors=pass_logits_proc,
                            prompt_cache=cache,
                            prefill_step_size=self.prefill_step_size,
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            quantized_kv_start=self.quantized_kv_start,
                        ):
                            if cancel_event is not None and cancel_event.is_set():
                                cancelled = True
                                break
                            tid = int(tok)
                            if tid in self.stop_tokens:
                                break
                            generated_ids.append(tid)
                            if mode != "TOOL_CALL":
                                raw_text = phase1_text + self.tokenizer.decode(
                                    generated_ids[phase1_len:]
                                )
                                yield from _drain_safe_text(raw_text)
                                if echo_detected:
                                    break
                mx.synchronize()
            finally:
                _ROT_CACHE["x_ref"] = None
                _ROT_CACHE["x_rot"] = None
                _ROT_CACHE["block"] = 0
                del cache

            t_end = time.perf_counter()
            if t_first is None:
                t_first = t_end

            raw_text = (
                (phase1_text + self.tokenizer.decode(generated_ids[phase1_len:]))
                if phase1_text is not None and phase1_len is not None
                else self.tokenizer.decode(generated_ids)
            )
            if not cancelled and not echo_detected and cursor < len(raw_text):
                if mode == "CONTENT":
                    tail_str = raw_text[cursor:]
                    if strip_leading_nl:
                        tail_str = tail_str.lstrip("\n")
                    if tail_str and "<tool_call>" not in tail_str and "<think>" not in tail_str:
                        yield from _emit_or_buffer_content(tail_str)
                elif mode == "THINK":
                    tail_r = raw_text[cursor:]
                    if "</think>" in tail_r:
                        r_part, c_part = tail_r.split("</think>", 1)
                        if r_part:
                            emitted_reasoning = True
                            yield {"type": "reasoning", "text": r_part}
                        c_part = c_part.lstrip("\n")
                        if c_part and "<tool_call>" not in c_part:
                            yield from _emit_or_buffer_content(c_part)
                    elif tail_r:
                        emitted_reasoning = True
                        yield {"type": "reasoning", "text": tail_r}

            if not cancelled and prev_assistant_text and not content_echo_checked and not echo_detected:
                if content_echo_buf:
                    joined_tail = "".join(content_echo_buf)
                    if is_verbatim_assistant_echo(joined_tail, prev_assistant_text, min_match_chars=60):
                        echo_detected = True
                    else:
                        content_echo_checked = True
                        content_echo_buf.clear()
                        if joined_tail.strip():
                            emitted_content = True
                        yield {"type": "content", "text": joined_tail}

            clean_content, reasoning_content, tool_calls = parse_model_output(raw_text, tools=tools)

            if not cancelled and prev_assistant_text and not echo_detected and clean_content:
                if not content_echo_checked and is_verbatim_assistant_echo(
                    clean_content, prev_assistant_text, min_match_chars=60
                ):
                    echo_detected = True

            if echo_detected and not cancelled:
                content_echo_buf.clear()
                retry_messages = copy.deepcopy(messages or [])
                anti_echo_note = (
                    "\n\n[SYSTEM ANTI-ECHO: Do NOT repeat or echo your previous response or menu. "
                    "Directly execute the user's selected option or perform the next concrete step now.]"
                )
                if retry_messages and isinstance(retry_messages[-1].get("content"), str):
                    retry_messages[-1]["content"] = retry_messages[-1]["content"] + anti_echo_note
                else:
                    retry_messages.append({"role": "user", "content": anti_echo_note.strip()})

                retry_prompt_str = self.render_prompt(
                    retry_messages,
                    tools=tools,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                )
                retry_prompt_ids = self.tokenizer.encode(
                    retry_prompt_str, add_special_tokens=False
                ).ids
                retry_len, retry_cache = self.prefix_cache.find_longest_prefix(
                    retry_prompt_ids, allow_turn_cache=True, consume_turn_entry=False
                )
                if retry_cache is None or retry_len >= len(retry_prompt_ids):
                    retry_cache = self.model.make_cache()
                    retry_len = 0
                retry_remaining_ids = retry_prompt_ids[retry_len:]
                if retry_remaining_ids:
                    retry_sampler = make_sampler(
                        temp=max(float(temperature), 0.75),
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                    )
                    retry_logits_proc = [
                        make_antiloop_logits_processor(
                            seed_token_ids=seed_token_ids,
                            protected_ids=getattr(self, "protected_syntax_ids", None),
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            repetition_penalty=max(float(repetition_penalty), 1.08),
                            context_size=384,
                            prose_mask=getattr(self, "_prose_mask", None),
                        )
                    ]
                    retry_ids: List[int] = []
                    try:
                        for tok, _ in generate_step(
                            mx.array(retry_remaining_ids),
                            self.model,
                            max_tokens=max_tokens,
                            sampler=retry_sampler,
                            logits_processors=retry_logits_proc,
                            prompt_cache=retry_cache,
                            prefill_step_size=self.prefill_step_size,
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            quantized_kv_start=self.quantized_kv_start,
                        ):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            tid = int(tok)
                            if tid in self.stop_tokens:
                                break
                            retry_ids.append(tid)
                        mx.synchronize()
                    finally:
                        _ROT_CACHE["x_ref"] = None
                        _ROT_CACHE["x_rot"] = None
                        _ROT_CACHE["block"] = 0
                        del retry_cache
                    if retry_ids:
                        generated_ids = retry_ids
                        t_end = time.perf_counter()
                        raw_text = self.tokenizer.decode(generated_ids)
                        clean_content, reasoning_content, tool_calls = parse_model_output(
                            raw_text, tools=tools
                        )
                        if reasoning_content and not emitted_reasoning:
                            emitted_reasoning = True
                            yield {"type": "reasoning", "text": reasoning_content}
                        if clean_content:
                            emitted_content = True
                            yield {"type": "content", "text": clean_content}

            if (
                not cancelled
                and not echo_detected
                and not is_aux
                and not emitted_content
                and not clean_content
                and not tool_calls
                and len(prompt_ids) >= 64
            ):
                retry_len, retry_cache = self.prefix_cache.find_longest_prefix(
                    prompt_ids, allow_turn_cache=False, consume_turn_entry=False
                )
                if retry_cache is None:
                    retry_cache = self.model.make_cache()
                    retry_len = 0
                retry_prompt_ids = prompt_ids[retry_len:]
                if retry_prompt_ids:
                    retry_sampler = make_sampler(
                        temp=max(temperature, 0.6), top_p=top_p, top_k=top_k, min_p=min_p
                    )
                    retry_logits_proc = [
                        make_antiloop_logits_processor(
                            seed_token_ids=seed_token_ids,
                            protected_ids=getattr(self, "protected_syntax_ids", None),
                            presence_penalty=max(float(presence_penalty), 1.2),
                            frequency_penalty=max(float(frequency_penalty), 0.25),
                            repetition_penalty=max(float(repetition_penalty), 1.06),
                            context_size=384,
                            prose_mask=getattr(self, "_prose_mask", None),
                        )
                    ]
                    retry_ids = []
                    try:
                        for tok, _ in generate_step(
                            mx.array(retry_prompt_ids),
                            self.model,
                            max_tokens=max_tokens,
                            sampler=retry_sampler,
                            logits_processors=retry_logits_proc,
                            prompt_cache=retry_cache,
                            prefill_step_size=self.prefill_step_size,
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            quantized_kv_start=self.quantized_kv_start,
                        ):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            tid = int(tok)
                            if tid in self.stop_tokens:
                                break
                            retry_ids.append(tid)
                        mx.synchronize()
                    finally:
                        _ROT_CACHE["x_ref"] = None
                        _ROT_CACHE["x_rot"] = None
                        _ROT_CACHE["block"] = 0
                        del retry_cache
                    if retry_ids:
                        generated_ids = retry_ids
                        t_end = time.perf_counter()
                        raw_text = self.tokenizer.decode(generated_ids)
                        clean_content, reasoning_content, tool_calls = parse_model_output(
                            raw_text, tools=tools
                        )
                        if reasoning_content and not emitted_reasoning:
                            yield {"type": "reasoning", "text": reasoning_content}
                        if clean_content:
                            yield {"type": "content", "text": clean_content}

            mx.clear_cache()

            ttft_s = t_first - t0
            decode_s = max(t_end - t_first, 1e-4)
            gen_tps = max(len(generated_ids) - 1, 1) / decode_s
            print(
                f"[NativeMLXEngine] prompt={len(prompt_ids)} (cached={initial_matched_len}, new={len(prompt_ids) - initial_matched_len}) "
                f"| gen={len(generated_ids)} | TTFT={ttft_s:.2f}s | decode={gen_tps:.1f} tok/s "
                f"| tool_calls={len(tool_calls)} | kv_bits={self.kv_bits}",
                flush=True,
            )

            yield {
                "type": "final",
                "result": {
                    "content": clean_content if clean_content or not tool_calls else None,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                    "prompt_tokens": len(prompt_ids),
                    "cached_tokens": initial_matched_len,
                    "completion_tokens": len(generated_ids),
                    "ttft_s": round(ttft_s, 3),
                    "decode_tok_s": round(gen_tps, 2),
                },
            }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "medium",
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        final_result: Optional[Dict[str, Any]] = None
        for event in self.generate_stream(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            cancel_event=cancel_event,
        ):
            if event.get("type") == "final":
                final_result = event["result"]
        if final_result is None:
            raise RuntimeError("generate_stream ended without final result")
        return final_result


class _BaseUpstreamHttpEngine:
    """Shared HTTP/SSE proxy & child-process supervisor for upstream inference servers."""

    backend: str = "upstream"
    _log_tag: str = "UpstreamEngine"
    _penalty_key: str = "repetition_penalty"

    def _init_upstream_base(
        self,
        *,
        upstream_host: str,
        upstream_port: int,
        context_length: int,
        model_id: str,
        arch: str,
        speculative: Optional[str] = None,
    ) -> None:
        self.lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.upstream_host = str(upstream_host or "127.0.0.1")
        self.upstream_port = int(upstream_port)
        self.context_length = int(context_length)
        self.model_id = str(model_id)
        self.arch = str(arch)
        self.kv_bits: Optional[int] = None
        self.kv_group_size: int = 64
        self.quantized_kv_start: int = 0
        self.prefill_step_size: int = 256
        self.wired_limit_mb: int = 0
        self.speculative: Optional[str] = speculative

        self_ref = weakref.ref(self)

        def _atexit_cleanup():
            inst = self_ref()
            if inst is not None:
                inst.shutdown()

        atexit.register(_atexit_cleanup)

    def _poll_upstream_health(
        self,
        *,
        process_label: str,
        startup_timeout_s: float = 120.0,
        allow_503: bool = False,
    ) -> None:
        deadline = time.monotonic() + float(startup_timeout_s)
        last_err: Optional[str] = None
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                rc = self.proc.returncode
                self.proc = None
                raise RuntimeError(f"{process_label} exited prematurely during startup with code {rc}")
            try:
                conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=2.0)
                try:
                    conn.request("GET", "/health")
                    resp = conn.getresponse()
                    raw = resp.read().decode("utf-8", errors="replace")
                    if resp.status == 200:
                        print(
                            f"[{self._log_tag}] Child {process_label} healthy on "
                            f"{self.upstream_host}:{self.upstream_port}: {raw.strip()}",
                            flush=True,
                        )
                        return
                    if allow_503 and resp.status == 503:
                        last_err = f"HTTP 503 (loading): {raw.strip()}"
                    else:
                        last_err = f"HTTP {resp.status}: {raw}"
                finally:
                    conn.close()
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.25)
        self.shutdown()
        raise RuntimeError(f"{process_label} did not become healthy within {startup_timeout_s}s ({last_err})")

    def shutdown(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
        except Exception:
            pass
        finally:
            self.proc = None

    def close(self) -> None:
        self.shutdown()

    def get_rss_gb(self) -> float:
        pids = [os.getpid()]
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None and proc.pid:
            pids.append(proc.pid)
        total_kb = 0
        try:
            res = subprocess.run(
                ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    s = line.strip()
                    if s.isdigit():
                        total_kb += int(s)
        except Exception:
            pass
        if total_kb <= 0:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            total_kb = int(ru // 1024) if sys.platform == "darwin" else int(ru)
        return round(total_kb / (1024 * 1024), 3)

    @staticmethod
    def _extract_prev_assistant_text(messages: Optional[List[Dict[str, Any]]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str) and len(re.sub(r"\s+", "", c)) >= 60:
                    return c
                break
        return ""

    @staticmethod
    def _normalize_stream_tool_calls(
        raw_accum: Dict[int, Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        tool_param_schemas: Dict[str, Dict[str, Any]] = {}
        if tools:
            for t in tools:
                fn = t.get("function", t)
                name = fn.get("name")
                props = fn.get("parameters", {}).get("properties", {})
                if name and isinstance(props, dict):
                    tool_param_schemas[name] = props

        result: List[Dict[str, Any]] = []
        for idx in sorted(raw_accum.keys()):
            entry = raw_accum[idx]
            fn_name = str(entry.get("name") or "").strip()
            if not fn_name:
                continue
            raw_args = entry.get("arguments", "")
            if isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                parsed = _repair_and_parse_json(str(raw_args or "{}"), expected_type="object")
                args_dict = parsed if isinstance(parsed, dict) else {}
            args_dict = _coerce_tool_args_dict(fn_name, args_dict, tool_param_schemas)
            result.append(
                {
                    "id": entry.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(args_dict, ensure_ascii=False),
                    },
                }
            )
        return result

    def _build_upstream_payload(
        self,
        *,
        norm_messages: List[Dict[str, Any]],
        compacted_tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        inkling_effort: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": norm_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            self._penalty_key: float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty),
            "reasoning_effort": inkling_effort,
            "chat_template_kwargs": {
                "reasoning_effort": inkling_effort,
                "enable_thinking": inkling_effort != "none",
            },
        }
        if compacted_tools:
            payload["tools"] = compacted_tools
        return payload

    def _run_upstream_sse_pass(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        inkling_effort: str,
        cancel_event: Optional[threading.Event],
    ) -> Iterator[Dict[str, Any]]:
        norm_messages = normalize_messages_for_template(
            messages,
            strip_historical_thinking=True,
            inject_loop_breaker=True,
            serialize_tool_args_for_jinja=False,
        )
        compacted_tools = compact_tools_for_template(tools)
        is_aux = not bool(tools)
        if is_aux:
            sys_blob = " ".join(str(m.get("content", "")) for m in norm_messages[:2])
            if "You are a summarization agent creating a context checkpoint" in sys_blob:
                max_tokens = min(max_tokens, 640)

        payload = self._build_upstream_payload(
            norm_messages=norm_messages,
            compacted_tools=compacted_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            inkling_effort=inkling_effort,
        )

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=300.0)
        done_event = threading.Event()
        sock_ref: List[Optional[socket.socket]] = [None]
        resp_ref: List[Optional[http.client.HTTPResponse]] = [None]

        def _cancel_watcher() -> None:
            while not done_event.is_set():
                if cancel_event is not None and cancel_event.wait(0.05):
                    try:
                        for s in (sock_ref[0], getattr(conn, "sock", None)):
                            if s is not None:
                                try:
                                    s.shutdown(socket.SHUT_RDWR)
                                except OSError:
                                    pass
                        if resp_ref[0] is not None:
                            try:
                                resp_ref[0].close()
                            except Exception:
                                pass
                        conn.close()
                    except Exception:
                        pass
                    return

        watcher_thread: Optional[threading.Thread] = None
        if cancel_event is not None:
            watcher_thread = threading.Thread(target=_cancel_watcher, daemon=True)
            watcher_thread.start()

        t0 = time.perf_counter()
        t_first: Optional[float] = None
        t_end: Optional[float] = None
        reasoning_parts: List[str] = []
        raw_content_parts: List[str] = []
        tool_accum: Dict[int, Dict[str, Any]] = {}
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        upstream_decode_tps: Optional[float] = None
        upstream_prompt_ms: Optional[float] = None

        try:
            if cancel_event is not None and cancel_event.is_set():
                yield {
                    "type": "pass_end",
                    "cancelled": True,
                    "reasoning": "",
                    "raw_content": "",
                    "tool_calls": [],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "ttft_s": 0.0,
                    "decode_tok_s": 0.0,
                }
                return

            conn.connect()
            sock_ref[0] = conn.sock
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body_bytes,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            resp = conn.getresponse()
            resp_ref[0] = resp
            if resp.status != 200:
                err_body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{self.backend} upstream returned HTTP {resp.status}: {err_body}")

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    raw_line = resp.fp.readline()
                except (OSError, ValueError, http.client.HTTPException):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    raise
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(chunk, dict) and chunk.get("error"):
                    err_obj = chunk["error"]
                    if isinstance(err_obj, dict):
                        err_msg = err_obj.get("message") or json.dumps(err_obj, ensure_ascii=False)
                    else:
                        err_msg = str(err_obj)
                    raise RuntimeError(f"{self.backend} upstream SSE error: {err_msg}")

                if isinstance(chunk.get("usage"), dict):
                    u = chunk["usage"]
                    prompt_tokens = int(u.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(u.get("completion_tokens") or completion_tokens)
                    details = u.get("prompt_tokens_details") or {}
                    if isinstance(details, dict) and "cached_tokens" in details:
                        cached_tokens = int(details["cached_tokens"] or 0)
                    if u.get("generation_tokens_per_second") is not None:
                        upstream_decode_tps = float(u["generation_tokens_per_second"])
                    if u.get("time_to_first_token") is not None:
                        upstream_prompt_ms = float(u["time_to_first_token"]) * 1000.0
                    elif u.get("prompt_eval_duration") is not None and upstream_prompt_ms is None:
                        upstream_prompt_ms = float(u["prompt_eval_duration"]) * 1000.0
                if isinstance(chunk.get("timings"), dict):
                    tm = chunk["timings"]
                    if tm.get("predicted_per_second") is not None:
                        upstream_decode_tps = float(tm["predicted_per_second"])
                    if tm.get("prompt_ms") is not None:
                        upstream_prompt_ms = float(tm["prompt_ms"])
                    if tm.get("cache_n") is not None and cached_tokens == 0:
                        cached_tokens = int(tm["cache_n"] or 0)
                    if tm.get("prompt_n") is not None and prompt_tokens == 0:
                        prompt_tokens = int(tm["prompt_n"] or 0)
                    if tm.get("predicted_n") is not None and completion_tokens == 0:
                        completion_tokens = int(tm["predicted_n"] or 0)

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason") == "error":
                    raise RuntimeError(
                        f"{self.backend} upstream SSE finished with error: {json.dumps(chunk, ensure_ascii=False)}"
                    )
                delta = choices[0].get("delta") or {}

                r_piece = delta.get("reasoning_content")
                if r_piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    reasoning_parts.append(r_piece)
                    yield {"type": "reasoning", "delta": r_piece, "text": r_piece}

                c_piece = delta.get("content")
                if c_piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    raw_content_parts.append(c_piece)
                    yield {"type": "content", "delta": c_piece, "text": c_piece}

                tc_list = delta.get("tool_calls")
                if isinstance(tc_list, list):
                    if t_first is None:
                        t_first = time.perf_counter()
                    for tc_delta in tc_list:
                        if not isinstance(tc_delta, dict):
                            continue
                        idx = int(tc_delta.get("index", 0))
                        slot = tool_accum.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                        if tc_delta.get("id"):
                            slot["id"] = tc_delta["id"]
                        fn_d = tc_delta.get("function") or {}
                        if fn_d.get("name"):
                            slot["name"] += str(fn_d["name"])
                        if fn_d.get("arguments") is not None:
                            slot["arguments"] += str(fn_d["arguments"])

            t_end = time.perf_counter()
        finally:
            done_event.set()
            try:
                conn.close()
            except Exception:
                pass
            if watcher_thread is not None:
                watcher_thread.join(timeout=0.5)

        cancelled = bool(cancel_event is not None and cancel_event.is_set())
        ttft_s = (
            round(upstream_prompt_ms / 1000.0, 3)
            if upstream_prompt_ms is not None
            else round((t_first - t0) if t_first is not None else (time.perf_counter() - t0), 3)
        )
        if upstream_decode_tps is not None:
            decode_tok_s = round(upstream_decode_tps, 2)
        elif t_first is not None and t_end is not None and completion_tokens > 0:
            decode_tok_s = round(max(completion_tokens - 1, 1) / max(t_end - t_first, 1e-4), 2)
        else:
            decode_tok_s = 0.0

        tool_calls = self._normalize_stream_tool_calls(tool_accum, tools=tools)
        yield {
            "type": "pass_end",
            "cancelled": cancelled,
            "reasoning": "".join(reasoning_parts),
            "raw_content": "".join(raw_content_parts),
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "ttft_s": ttft_s,
            "decode_tok_s": decode_tok_s,
        }

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "low",
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        with self.lock:
            yield from self._generate_stream_unlocked(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                cancel_event=cancel_event,
            )

    def _generate_stream_unlocked(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "low",
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        if temperature is None:
            temperature = float(os.environ.get("M5_MLX_DEFAULT_TEMP", "0.6"))
        if top_p is None:
            top_p = float(os.environ.get("M5_MLX_DEFAULT_TOP_P", "0.95"))
        if presence_penalty is None:
            presence_penalty = 0.0
        if frequency_penalty is None:
            frequency_penalty = 0.0

        loop_info = detect_semantic_tool_loop(messages)
        temperature, repetition_penalty = resolve_sampling_with_loop_guard(
            messages, temperature, repetition_penalty
        )
        if loop_info.get("is_loop"):
            temperature = max(float(temperature), 0.9)
            presence_penalty = max(float(presence_penalty), 1.5)
            frequency_penalty = max(float(frequency_penalty), 0.4)
            repetition_penalty = max(float(repetition_penalty), 1.08)

        inkling_effort = resolve_inkling_reasoning_effort(
            reasoning_effort, enable_thinking=enable_thinking
        )
        prev_assistant_text = self._extract_prev_assistant_text(messages)
        pass_cancel = threading.Event()
        if cancel_event is not None and cancel_event.is_set():
            pass_cancel.set()

        content_echo_buf: List[str] = []
        content_echo_checked = not bool(prev_assistant_text)
        echo_detected = False
        emitted_reasoning = False
        emitted_content = False
        pass_summary: Optional[Dict[str, Any]] = None

        def _relay_pass(
            pass_messages: List[Dict[str, Any]],
            pass_temp: float,
            pass_rep: float,
            check_echo: bool,
        ) -> Iterator[Dict[str, Any]]:
            nonlocal echo_detected, content_echo_checked, emitted_reasoning, emitted_content, pass_summary
            local_cancel = cancel_event
            if check_echo:
                local_cancel = threading.Event()
                if cancel_event is not None and cancel_event.is_set():
                    local_cancel.set()

            def _forward_external_cancel():
                if cancel_event is not None and local_cancel is not None and local_cancel is not cancel_event:
                    while not local_cancel.is_set():
                        if cancel_event.wait(0.05):
                            local_cancel.set()
                            return

            fwd_thread: Optional[threading.Thread] = None
            if check_echo and cancel_event is not None:
                fwd_thread = threading.Thread(target=_forward_external_cancel, daemon=True)
                fwd_thread.start()

            try:
                for ev in self._run_upstream_sse_pass(
                    messages=pass_messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=pass_temp,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=pass_rep,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    inkling_effort=inkling_effort,
                    cancel_event=local_cancel,
                ):
                    etype = ev["type"]
                    if etype in ("reasoning", "reasoning_piece"):
                        r_txt = ev.get("text") if "text" in ev else ev.get("delta", "")
                        emitted_reasoning = True
                        yield {"type": "reasoning", "text": r_txt}
                    elif etype in ("content", "content_piece"):
                        raw_txt = ev.get("text") if "text" in ev else ev.get("delta", "")
                        piece = str(raw_txt).replace("<|content_model_end_sampling|>", "")
                        if not piece:
                            continue
                        if not check_echo or content_echo_checked:
                            if piece.strip():
                                emitted_content = True
                            yield {"type": "content", "text": piece}
                        else:
                            content_echo_buf.append(piece)
                            joined = "".join(content_echo_buf)
                            non_ws_len = len(re.sub(r"\s+", "", joined))
                            if len(joined) >= 72 and non_ws_len >= 60:
                                if is_verbatim_assistant_echo(joined, prev_assistant_text, min_match_chars=60):
                                    echo_detected = True
                                    if local_cancel is not None:
                                        local_cancel.set()
                                    continue
                                content_echo_checked = True
                                content_echo_buf.clear()
                                if joined.strip():
                                    emitted_content = True
                                yield {"type": "content", "text": joined}
                    elif etype == "pass_end":
                        pass_summary = ev
            finally:
                if local_cancel is not None and local_cancel is not cancel_event:
                    local_cancel.set()
                if fwd_thread is not None:
                    fwd_thread.join(timeout=0.5)

        yield from _relay_pass(messages, temperature, repetition_penalty, bool(prev_assistant_text))

        cancelled = bool(cancel_event is not None and cancel_event.is_set())
        if not cancelled and not echo_detected and prev_assistant_text and not content_echo_checked and content_echo_buf:
            joined_tail = "".join(content_echo_buf)
            if is_verbatim_assistant_echo(joined_tail, prev_assistant_text, min_match_chars=60):
                echo_detected = True
            else:
                content_echo_checked = True
                content_echo_buf.clear()
                if joined_tail.strip():
                    emitted_content = True
                yield {"type": "content", "text": joined_tail}

        if echo_detected and not cancelled:
            content_echo_buf.clear()
            content_echo_checked = True
            retry_messages = copy.deepcopy(messages or [])
            anti_echo_note = (
                "\n\n[SYSTEM ANTI-ECHO: Do NOT repeat or echo your previous response or menu. "
                "Directly execute the user's selected option or perform the next concrete step now.]"
            )
            if retry_messages and isinstance(retry_messages[-1].get("content"), str):
                retry_messages[-1]["content"] = retry_messages[-1]["content"] + anti_echo_note
            else:
                retry_messages.append({"role": "user", "content": anti_echo_note.strip()})
            yield from _relay_pass(
                retry_messages,
                max(float(temperature), 0.75),
                max(float(repetition_penalty), 1.08),
                False,
            )
            cancelled = bool(cancel_event is not None and cancel_event.is_set())

        if pass_summary is None:
            pass_summary = {
                "reasoning": "",
                "raw_content": "",
                "tool_calls": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "ttft_s": 0.0,
                "decode_tok_s": 0.0,
            }

        raw_content = pass_summary["raw_content"]
        reasoning_text = pass_summary["reasoning"].strip() or None
        tool_calls = list(pass_summary["tool_calls"])

        clean_content = raw_content.replace("<|content_model_end_sampling|>", "").strip()
        if (
            "<|content_" in raw_content
            or "<|message_model|>" in raw_content
            or "<tool_call>" in raw_content
        ):
            parsed_c, parsed_r, parsed_tcs = parse_model_output(raw_content, tools=tools)
            clean_content = parsed_c
            if parsed_r and not reasoning_text:
                reasoning_text = parsed_r
                if not emitted_reasoning:
                    yield {"type": "reasoning", "text": reasoning_text}
            if parsed_tcs and not tool_calls:
                tool_calls = parsed_tcs

        print(
            f"[{self._log_tag}] prompt={pass_summary['prompt_tokens']} "
            f"(cached={pass_summary['cached_tokens']}) | gen={pass_summary['completion_tokens']} "
            f"| TTFT={pass_summary['ttft_s']:.2f}s | decode={pass_summary['decode_tok_s']:.1f} tok/s "
            f"| tool_calls={len(tool_calls)} | spec={self.speculative}",
            flush=True,
        )

        yield {
            "type": "final",
            "result": {
                "content": clean_content if clean_content or not tool_calls else None,
                "reasoning_content": reasoning_text,
                "tool_calls": tool_calls,
                "prompt_tokens": pass_summary["prompt_tokens"],
                "cached_tokens": pass_summary["cached_tokens"],
                "completion_tokens": pass_summary["completion_tokens"],
                "ttft_s": pass_summary["ttft_s"],
                "decode_tok_s": pass_summary["decode_tok_s"],
            },
        }

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "low",
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        final_result: Optional[Dict[str, Any]] = None
        for event in self.generate_stream(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            cancel_event=cancel_event,
        ):
            if event.get("type") == "final":
                final_result = event["result"]
        if final_result is None:
            raise RuntimeError("generate_stream ended without final result")
        return final_result


class LlamaCppBackendEngine(_BaseUpstreamHttpEngine):
    """Supervises Unsloth llama-server over localhost HTTP/SSE without allocating MLX RAM."""

    backend = "llama-cpp"
    _log_tag = "LlamaCppBackendEngine"
    _penalty_key = "repeat_penalty"

    def __init__(
        self,
        *,
        llama_server_bin: Optional[Path] = None,
        model_file: Optional[Path] = None,
        draft_model_file: Optional[Path] = None,
        context_length: int = 131072,
        llama_host: str = "127.0.0.1",
        llama_port: int = 8081,
        spec_draft_max: int = 0,
        kv_cache_type: str = "q4_0",
        threads: Optional[int] = None,
        model_id: Optional[str] = None,
        arch: str = "inkling2_loop",
        speculative: Optional[str] = None,
        start_subprocess: bool = True,
        startup_timeout_s: float = 120.0,
    ):
        self.llama_host = str(llama_host or "127.0.0.1")
        self.llama_port = int(llama_port)
        self.spec_draft_max = int(spec_draft_max)
        self.kv_cache_type = str(kv_cache_type or "q4_0").strip().lower()
        self.threads = int(threads) if threads else None
        self.llama_server_bin = Path(llama_server_bin) if llama_server_bin else None
        self.model_file = Path(model_file) if model_file else None
        self.draft_model_file = Path(draft_model_file) if draft_model_file else None

        validate_backend_config(
            backend="llama-cpp",
            llama_server_bin=self.llama_server_bin,
            model_file=self.model_file,
            draft_model_file=self.draft_model_file,
            llama_port=self.llama_port,
            spec_draft_max=self.spec_draft_max,
            kv_cache_type=self.kv_cache_type,
            require_paths=start_subprocess,
        )

        if speculative is not None:
            resolved_spec: Optional[str] = speculative
        elif self.draft_model_file is not None and self.spec_draft_max > 0:
            resolved_spec = "draft-dspark"
        else:
            resolved_spec = None

        self._init_upstream_base(
            upstream_host=self.llama_host,
            upstream_port=self.llama_port,
            context_length=int(context_length),
            model_id=model_id or os.environ.get("M5_MLX_MODEL_ID", "inkling-2-mini-ud-q3-k-xl"),
            arch=arch,
            speculative=resolved_spec,
        )

        if start_subprocess:
            self._start_llama_server(startup_timeout_s=startup_timeout_s)

    def _build_llama_server_cmd(self) -> List[str]:
        cmd = [
            str(self.llama_server_bin),
            "-m",
            str(self.model_file),
            "--host",
            self.llama_host,
            "--port",
            str(self.llama_port),
            "-c",
            str(self.context_length),
            "-ngl",
            "99",
            "-fa",
            "on",
            "-np",
            "1",
            "-b",
            "512",
            "-ub",
            "64",
            "--cache-reuse",
            "256",
            "--jinja",
            "--metrics",
        ]
        if self.threads:
            cmd.extend(["-t", str(self.threads)])
        if self.kv_cache_type and self.kv_cache_type not in ("f16", "bf16"):
            cmd.extend(["-ctk", self.kv_cache_type, "-ctv", self.kv_cache_type])
        if self.draft_model_file and self.spec_draft_max > 0:
            cmd.extend(
                [
                    "-md",
                    str(self.draft_model_file),
                    "--spec-type",
                    "draft-dspark",
                    "--spec-draft-n-max",
                    str(self.spec_draft_max),
                ]
            )
        return cmd

    def _start_llama_server(self, startup_timeout_s: float = 120.0) -> None:
        cmd = self._build_llama_server_cmd()
        print(f"[LlamaCppBackendEngine] Starting child llama-server: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=None, stderr=None)
        self._poll_upstream_health(
            process_label="llama-server",
            startup_timeout_s=startup_timeout_s,
            allow_503=False,
        )


class OmlxBackendEngine(_BaseUpstreamHttpEngine):
    """Supervises oMLX 0.8.2 (`omlx serve`) over localhost HTTP/SSE with serialized sticky llama-server overflow."""

    _log_tag = "OmlxBackendEngine"
    _penalty_key = "repetition_penalty"
    _PREFILL_GUARD_MARKERS = (
        "prefill memory guard",
        "prefill_memory_exceeded",
        "prefill_memory_aborted",
        "prefill context too large",
        "memory limit exceeded",
        "http 507",
    )

    def __init__(
        self,
        *,
        omlx_bin: Optional[Path] = None,
        omlx_model_dir: Optional[Path] = None,
        omlx_data_dir: Optional[Path] = None,
        omlx_ssd_cache_dir: Optional[Path] = None,
        omlx_host: str = "127.0.0.1",
        omlx_port: int = 8081,
        context_length: int = 131072,
        model_id: Optional[str] = None,
        arch: str = "inkling2_loop",
        speculative: Optional[str] = None,
        start_subprocess: bool = True,
        startup_timeout_s: float = 180.0,
        overflow_llama_bin: Optional[Path] = None,
        overflow_model_file: Optional[Path] = None,
        overflow_tokens_threshold: int = 16500,
        restore_idle_s: float = 300.0,
        restore_max_tokens: int = 4000,
    ):
        m5_home = _get_m5_home()
        self._m5_home = m5_home
        self.omlx_host = str(omlx_host or "127.0.0.1")
        self.omlx_port = int(omlx_port)
        self.omlx_bin = Path(omlx_bin) if omlx_bin else (m5_home / "omlx-env" / "bin" / "omlx")
        self.omlx_model_dir = Path(omlx_model_dir) if omlx_model_dir else (m5_home / "omlx-models")
        self.omlx_data_dir = Path(omlx_data_dir) if omlx_data_dir else (m5_home / "omlx-data")
        self.omlx_ssd_cache_dir = Path(omlx_ssd_cache_dir) if omlx_ssd_cache_dir else (m5_home / "omlx-ssd-cache")
        self.start_subprocess = bool(start_subprocess)
        self.startup_timeout_s = float(startup_timeout_s)
        self._custom_overflow_llama_bin = Path(overflow_llama_bin) if overflow_llama_bin else None
        self._custom_overflow_model_file = Path(overflow_model_file) if overflow_model_file else None
        self._overflow_tokens_threshold = int(overflow_tokens_threshold)
        self._restore_idle_s = float(restore_idle_s)
        self._restore_max_tokens = int(restore_max_tokens)
        self._overflow_engine: Optional[LlamaCppBackendEngine] = None
        self._last_activity_ts: float = time.monotonic()
        resolved_model_id = model_id or os.environ.get("M5_MLX_MODEL_ID", "inkling-2-mini-oq3e-mtp")

        validate_backend_config(
            backend="omlx",
            omlx_bin=self.omlx_bin,
            omlx_model_dir=self.omlx_model_dir,
            omlx_port=self.omlx_port,
            model_id=resolved_model_id,
            require_paths=start_subprocess,
        )

        self._init_upstream_base(
            upstream_host=self.omlx_host,
            upstream_port=self.omlx_port,
            context_length=int(context_length),
            model_id=resolved_model_id,
            arch=arch,
            speculative=speculative,
        )

        if start_subprocess:
            self._start_omlx_server(startup_timeout_s=startup_timeout_s)

    @property
    def backend(self) -> str:
        overflow = getattr(self, "_overflow_engine", None)
        if overflow is not None:
            return overflow.backend
        return "omlx"

    def shutdown(self) -> None:
        overflow = getattr(self, "_overflow_engine", None)
        if overflow is not None:
            try:
                overflow.shutdown()
            finally:
                self._overflow_engine = None
        super().shutdown()

    def get_rss_gb(self) -> float:
        overflow = getattr(self, "_overflow_engine", None)
        if overflow is not None:
            return overflow.get_rss_gb()
        return super().get_rss_gb()

    @staticmethod
    def _resolve_llama_server_bin(m5_home: Path) -> Path:
        candidates = [
            m5_home / "bin" / "llama-server",
            m5_home / "bin" / "unsloth-llama-cpp" / "llama-server",
            m5_home / "bin" / "unsloth-llama-cpp" / "llama-b12588-mix-c0c3de8" / "llama-server",
            m5_home / "llama.cpp" / "llama-server",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        unsloth_root = m5_home / "bin" / "unsloth-llama-cpp"
        if unsloth_root.is_dir():
            for cand in sorted(unsloth_root.glob("**/llama-server")):
                if cand.is_file():
                    return cand
        return m5_home / "bin" / "llama-server"

    def _get_overflow_paths(self) -> Tuple[Path, Path]:
        raw_home = os.environ.get("M5_MLX_HOME", "").strip()
        m5_home = Path(raw_home) if raw_home else self._m5_home
        if self._custom_overflow_llama_bin is not None:
            llama_bin = self._custom_overflow_llama_bin
        else:
            env_bin = os.environ.get("M5_MLX_OVERFLOW_LLAMA_BIN", "").strip()
            llama_bin = Path(env_bin) if env_bin else self._resolve_llama_server_bin(m5_home)
        if self._custom_overflow_model_file is not None:
            model_file = self._custom_overflow_model_file
        else:
            env_model = os.environ.get("M5_MLX_OVERFLOW_MODEL_FILE", "").strip()
            model_file = (
                Path(env_model)
                if env_model
                else (m5_home / "models" / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q3_K_XL.gguf")
            )
        return llama_bin, model_file

    def _can_overflow(self) -> bool:
        llama_bin, model_file = self._get_overflow_paths()
        return llama_bin.is_file() and model_file.is_file()

    @staticmethod
    def _wait_for_port_release(host: str, port: int, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, int(port)), timeout=0.15):
                    pass
            except OSError:
                return
            time.sleep(0.05)

    @classmethod
    def _is_prefill_memory_guard_error(cls, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(marker in msg for marker in cls._PREFILL_GUARD_MARKERS)

    def _promote_to_overflow_locked(self, reason: str) -> None:
        llama_bin, model_file = self._get_overflow_paths()
        if not (llama_bin.is_file() and model_file.is_file()):
            raise FileNotFoundError(f"Overflow backend unavailable (bin={llama_bin}, model={model_file})")
        print(
            f"[OmlxBackendEngine] Promoting to Unsloth llama-server overflow ({reason}); "
            f"stopping omlx on {self.omlx_host}:{self.omlx_port} first",
            flush=True,
        )
        self.shutdown()
        if self.start_subprocess:
            self._wait_for_port_release(self.omlx_host, self.omlx_port, timeout_s=10.0)
        self._overflow_engine = LlamaCppBackendEngine(
            llama_server_bin=llama_bin,
            model_file=model_file,
            draft_model_file=None,
            context_length=131072,
            llama_host=self.omlx_host,
            llama_port=self.omlx_port,
            spec_draft_max=0,
            kv_cache_type="q4_0",
            threads=6,
            model_id=self.model_id,
            arch=self.arch,
            speculative=None,
            start_subprocess=self.start_subprocess,
            startup_timeout_s=self.startup_timeout_s,
        )

    def _restore_omlx_locked(self) -> None:
        print(
            f"[OmlxBackendEngine] Idle short-context restore triggered; "
            f"stopping llama-server on {self.omlx_host}:{self.omlx_port} before restarting omlx",
            flush=True,
        )
        if self._overflow_engine is not None:
            try:
                self._overflow_engine.shutdown()
            finally:
                self._overflow_engine = None
        if self.start_subprocess:
            self._wait_for_port_release(self.omlx_host, self.omlx_port, timeout_s=10.0)
            self._start_omlx_server(startup_timeout_s=self.startup_timeout_s)

    def write_model_settings(self) -> Path:
        """Atomically persist v2 model_settings.json compatible with oMLX ModelSettingsManager."""
        self.omlx_data_dir.mkdir(parents=True, exist_ok=True)
        settings_path = self.omlx_data_dir / "model_settings.json"

        data: Dict[str, Any] = {"version": 2, "models": {}}
        if settings_path.exists():
            try:
                loaded = json.loads(settings_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
                    data["models"] = loaded["models"]
            except (OSError, ValueError):
                pass

        models = data["models"]
        for mid, entry in list(models.items()):
            if isinstance(entry, dict) and mid != self.model_id:
                entry["is_default"] = False

        existing = dict(models.get(self.model_id) or {}) if isinstance(models.get(self.model_id), dict) else {}
        existing.update(
            {
                "max_context_window": int(self.context_length),
                "mtp_enabled": False,
                "dflash_enabled": False,
                "vlm_mtp_enabled": False,
                "is_pinned": True,
                "is_default": True,
            }
        )
        models[self.model_id] = existing
        data["version"] = 2

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.omlx_data_dir,
                prefix=".model_settings.",
                suffix=".tmp",
                delete=False,
            ) as tmp_fh:
                temp_path = Path(tmp_fh.name)
                json.dump(data, tmp_fh, indent=2, ensure_ascii=False)
                tmp_fh.flush()
                os.fsync(tmp_fh.fileno())
            os.replace(temp_path, settings_path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        return settings_path

    def _build_omlx_server_cmd(self) -> List[str]:
        cmd = [
            str(self.omlx_bin),
            "serve",
            "--model-dir",
            str(self.omlx_model_dir),
            "--base-path",
            str(self.omlx_data_dir),
            "--host",
            self.omlx_host,
            "--port",
            str(self.omlx_port),
            "--headless",
            "--no-hf-cache",
            "--sse-keepalive-mode",
            "chunk",
        ]
        raw_gb = os.environ.get("M5_MLX_OMLX_MEMORY_GUARD_GB")
        raw_tier = os.environ.get("M5_MLX_OMLX_MEMORY_GUARD")
        if raw_gb is not None and raw_gb.strip():
            cmd.extend(["--memory-guard-gb", raw_gb.strip()])
        elif raw_tier is not None and raw_tier.strip():
            cmd.extend(["--memory-guard", raw_tier.strip()])
        else:
            cmd.extend(["--memory-guard-gb", "20.0"])
        cmd.extend(
            [
                "--paged-ssd-cache-dir",
                str(self.omlx_ssd_cache_dir),
                "--paged-ssd-cache-max-size",
                "20GB",
                "--hot-cache-max-size",
                str(os.environ.get("M5_MLX_OMLX_HOT_CACHE", "0")),
                "--hot-cache-write-through",
                "--max-concurrent-requests",
                str(os.environ.get("M5_MLX_OMLX_MAX_CONCURRENT", "1")),
            ]
        )
        return cmd

    def _warm_model_via_http(self, timeout_s: float = 120.0) -> None:
        encoded_id = urllib.parse.quote(self.model_id, safe="")
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=timeout_s)
        try:
            conn.request("POST", f"/v1/models/{encoded_id}/load", body=b"", headers={"Content-Length": "0"})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"oMLX warmup /v1/models/{encoded_id}/load failed HTTP {resp.status}: {raw}")
            print(
                f"[OmlxBackendEngine] Model {self.model_id} loaded/warmed via /v1/models/{encoded_id}/load: {raw.strip()}",
                flush=True,
            )
        finally:
            conn.close()

    def _start_omlx_server(self, startup_timeout_s: float = 180.0) -> None:
        self.write_model_settings()
        self.omlx_ssd_cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_omlx_server_cmd()
        print(f"[OmlxBackendEngine] Starting child omlx serve: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, stdout=None, stderr=None)
        self._poll_upstream_health(
            process_label="omlx",
            startup_timeout_s=startup_timeout_s,
            allow_503=True,
        )
        self._warm_model_via_http(timeout_s=min(float(startup_timeout_s), 120.0))

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.03,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        reasoning_effort: str = "low",
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        with self.lock:
            now = time.monotonic()
            idle_elapsed = now - self._last_activity_ts
            self._last_activity_ts = now

            norm_messages = normalize_messages_for_template(
                messages,
                strip_historical_thinking=True,
                inject_loop_breaker=True,
                serialize_tool_args_for_jinja=False,
            )
            compacted_tools = compact_tools_for_template(tools)
            est_tokens = _estimate_messages_tokens(norm_messages, compacted_tools)

            raw_ov_tok = os.environ.get("M5_MLX_OVERFLOW_TOKENS", "").strip()
            overflow_threshold = int(raw_ov_tok) if raw_ov_tok else self._overflow_tokens_threshold
            raw_idle_s = os.environ.get("M5_MLX_RESTORE_IDLE_S", "").strip()
            restore_idle_s = float(raw_idle_s) if raw_idle_s else self._restore_idle_s

            if self._overflow_engine is not None:
                if est_tokens < self._restore_max_tokens and idle_elapsed >= restore_idle_s:
                    self._restore_omlx_locked()

            if self._overflow_engine is None and est_tokens >= overflow_threshold and self._can_overflow():
                self._promote_to_overflow_locked(
                    reason=f"proactive est_tokens={est_tokens} >= {overflow_threshold}"
                )

            try:
                if self._overflow_engine is not None:
                    yield from self._overflow_engine.generate_stream(
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        cancel_event=cancel_event,
                    )
                    return

                emitted_any = False
                try:
                    for ev in self._generate_stream_unlocked(
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                        cancel_event=cancel_event,
                    ):
                        if ev.get("type") in ("reasoning", "content", "final"):
                            emitted_any = True
                        yield ev
                except RuntimeError as exc:
                    if (
                        not emitted_any
                        and self._is_prefill_memory_guard_error(exc)
                        and self._can_overflow()
                        and (cancel_event is None or not cancel_event.is_set())
                    ):
                        self._promote_to_overflow_locked(reason=f"reactive: {exc}")
                        assert self._overflow_engine is not None
                        yield from self._overflow_engine.generate_stream(
                            messages=messages,
                            tools=tools,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            repetition_penalty=repetition_penalty,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            enable_thinking=enable_thinking,
                            reasoning_effort=reasoning_effort,
                            cancel_event=cancel_event,
                        )
                        return
                    raise
            finally:
                self._last_activity_ts = time.monotonic()


app = FastAPI(title="Apple M5 MLX OpenAI Server")
ENGINE: Optional[Any] = None


def _cleanup_on_exit(*_args) -> None:
    global ENGINE, _SERVER_LOCK_FH
    eng = ENGINE
    ENGINE = None
    if eng is not None:
        try:
            if hasattr(eng, "close"):
                eng.close()
            elif hasattr(eng, "shutdown"):
                eng.shutdown()
        except Exception:
            pass
    fh = _SERVER_LOCK_FH
    _SERVER_LOCK_FH = None
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    global ENGINE

    def _init_and_warm():
        backend = _env_str("M5_MLX_BACKEND", "omlx").lower()
        validate_backend_config(backend=backend)
        if backend == "omlx":
            m5_home = _get_m5_home()
            return OmlxBackendEngine(
                omlx_bin=Path(_env_str("M5_MLX_OMLX_BIN", str(m5_home / "omlx-env" / "bin" / "omlx"))),
                omlx_model_dir=Path(_env_str("M5_MLX_OMLX_MODEL_DIR", str(m5_home / "omlx-models"))),
                omlx_data_dir=Path(_env_str("M5_MLX_OMLX_DATA_DIR", str(m5_home / "omlx-data"))),
                omlx_ssd_cache_dir=Path(_env_str("M5_MLX_OMLX_SSD_CACHE_DIR", str(m5_home / "omlx-ssd-cache"))),
                context_length=_env_int("M5_MLX_CONTEXT_LENGTH", 131072),
                omlx_port=_env_int("M5_MLX_OMLX_PORT", 8081),
                model_id=_env_str("M5_MLX_MODEL_ID", "inkling-2-mini-oq3e-mtp"),
                start_subprocess=True,
            )
        if backend == "llama-cpp":
            draft_raw = _env_str("M5_MLX_DRAFT_MODEL_FILE", "")
            threads_raw = _env_str("M5_MLX_LLAMA_THREADS", "")
            return LlamaCppBackendEngine(
                llama_server_bin=Path(_env_str("M5_MLX_LLAMA_SERVER_BIN", "")),
                model_file=Path(_env_str("M5_MLX_MODEL_FILE", "")),
                draft_model_file=Path(draft_raw) if draft_raw else None,
                context_length=_env_int("M5_MLX_CONTEXT_LENGTH", 131072),
                llama_port=_env_int("M5_MLX_LLAMA_PORT", 8081),
                spec_draft_max=_env_int("M5_MLX_SPEC_DRAFT_MAX", 0),
                kv_cache_type=_env_str("M5_MLX_KV_CACHE_TYPE", "q4_0"),
                threads=int(threads_raw) if threads_raw else None,
                model_id=_env_str("M5_MLX_MODEL_ID", "inkling-2-mini-ud-q3-k-xl"),
                start_subprocess=True,
            )
        default_kv = _env_str("M5_MLX_KV_BITS", "4")
        model_dir = Path(_env_str("M5_MLX_MODEL_DIR", str(PACK_DIR)))
        return NativeMLXEngine(model_dir, kv_bits=default_kv)  # type: ignore[arg-type]

    loop = asyncio.get_running_loop()
    ENGINE = await loop.run_in_executor(_MLX_EXECUTOR, _init_and_warm)


@app.on_event("shutdown")
async def shutdown_event():
    _cleanup_on_exit()


def _process_rss_gb() -> float:
    if ENGINE is not None and hasattr(ENGINE, "get_rss_gb"):
        return float(ENGINE.get_rss_gb())
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kb = int(ru // 1024) if sys.platform == "darwin" else int(ru)
    return round(kb / (1024 * 1024), 3)


def _owned_by_for_backend(backend: str) -> str:
    if backend == "omlx":
        return "m5-omlx"
    if backend == "llama-cpp":
        return "m5-llama-cpp"
    return "m5-mlx"


@app.get("/health")
def health():
    backend = getattr(ENGINE, "backend", os.environ.get("M5_MLX_BACKEND", "omlx")) if ENGINE else os.environ.get("M5_MLX_BACKEND", "omlx")
    is_mlx = backend == "mlx"
    default_model_id = (
        "inkling-2-mini-oq3e-mtp"
        if backend == "omlx"
        else ("inkling-2-mini-ud-q3-k-xl" if backend == "llama-cpp" else "qwen3.6-35b-a3b-3bit-mlx")
    )
    return {
        "status": "ok",
        "backend": backend,
        "model": ENGINE.model_id if ENGINE else os.environ.get("M5_MLX_MODEL_ID", default_model_id),
        "arch": ENGINE.arch if ENGINE else ("inkling2_loop" if not is_mlx else "unknown"),
        "speculative": getattr(ENGINE, "speculative", None) if ENGINE else None,
        "rss_gb": _process_rss_gb(),
        "kv_bits": ENGINE.kv_bits if ENGINE else None,
        "kv_group_size": ENGINE.kv_group_size if ENGINE else 64,
        "quantized_kv_start": ENGINE.quantized_kv_start if ENGINE else 0,
        "context_length": ENGINE.context_length if ENGINE else int(os.environ.get("M5_MLX_CONTEXT_LENGTH", "131072")),
        "prefill_step_size": ENGINE.prefill_step_size if ENGINE else int(os.environ.get("M5_MLX_PREFILL_STEP_SIZE", "256")),
        "wired_limit_mb": getattr(ENGINE, "wired_limit_mb", 0) if ENGINE else (int(os.environ.get("M5_MLX_WIRED_LIMIT_MB", "6144")) if is_mlx else 0),
        "active_mem_gb": round(mx.get_active_memory() / (1024**3), 3) if is_mlx else 0.0,
        "cache_mem_gb": round(mx.get_cache_memory() / (1024**3), 3) if is_mlx else 0.0,
        "peak_mem_gb": round(mx.get_peak_memory() / (1024**3), 3) if is_mlx else 0.0,
    }


@app.get("/v1/models")
@app.get("/api/v1/models")
def list_models():
    now = int(time.time())
    ctx_len = ENGINE.context_length if ENGINE else int(os.environ.get("M5_MLX_CONTEXT_LENGTH", "131072"))
    backend = getattr(ENGINE, "backend", os.environ.get("M5_MLX_BACKEND", "omlx")) if ENGINE else os.environ.get("M5_MLX_BACKEND", "omlx")
    default_model_id = (
        "inkling-2-mini-oq3e-mtp"
        if backend == "omlx"
        else ("inkling-2-mini-ud-q3-k-xl" if backend == "llama-cpp" else "qwen3.6-35b-a3b-3bit-mlx")
    )
    primary_id = ENGINE.model_id if ENGINE else os.environ.get("M5_MLX_MODEL_ID", default_model_id)
    owned_by = _owned_by_for_backend(backend)
    ids = [
        primary_id,
        "inkling-2-mini-oq3e-mtp",
        "inkling-2-mini-ud-q3-k-xl",
        "inkling-2-mini",
        "unsloth/Inkling-2-Mini-GGUF",
        "qwen3.6-35b-a3b-3bit-mlx",
        "qwen3.6-35b-a3b-nvfp4-mlx",
        "qwen3.6-35b-a3b-nvfp4",
    ]
    seen = set()
    data = []
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        data.append(
            {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": owned_by,
                "context_length": ctx_len,
                "max_model_len": ctx_len,
                "loaded_context_length": ctx_len,
            }
        )
    return {
        "object": "list",
        "data": data,
    }


@app.get("/v1/models/{model_id:path}")
@app.get("/api/v1/models/{model_id:path}")
def get_model(model_id: str):
    now = int(time.time())
    ctx_len = ENGINE.context_length if ENGINE else int(os.environ.get("M5_MLX_CONTEXT_LENGTH", "131072"))
    backend = getattr(ENGINE, "backend", os.environ.get("M5_MLX_BACKEND", "omlx")) if ENGINE else os.environ.get("M5_MLX_BACKEND", "omlx")
    owned_by = _owned_by_for_backend(backend)
    return {
        "id": model_id,
        "object": "model",
        "created": now,
        "owned_by": owned_by,
        "context_length": ctx_len,
        "max_model_len": ctx_len,
        "loaded_context_length": ctx_len,
    }


@app.get("/v1/props")
@app.get("/props")
def server_props():
    ctx_len = ENGINE.context_length if ENGINE else int(os.environ.get("M5_MLX_CONTEXT_LENGTH", "131072"))
    return {
        "default_generation_settings": {
            "n_ctx": ctx_len,
        },
        "total_slots": 1,
    }


@app.get("/api/tags")
def ollama_tags():
    return {
        "models": [
            {
                "name": "inkling-2-mini-oq3e-mtp",
                "model": "inkling-2-mini-oq3e-mtp",
            },
            {
                "name": "qwen3.6-35b-a3b-3bit-mlx",
                "model": "qwen3.6-35b-a3b-3bit-mlx",
            },
        ]
    }


@app.get("/version")
def server_version():
    return {"version": "0.32.4-m5-mlx"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if ENGINE is None:
        return JSONResponse({"error": "Engine warming up"}, status_code=503)
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
    model_name = body.get("model", "inkling-2-mini-oq3e-mtp")

    ct_kwargs = body.get("chat_template_kwargs") or {}
    req_reasoning = body.get("reasoning_effort") or ct_kwargs.get("reasoning_effort")
    is_inkling_upstream = getattr(ENGINE, "backend", "mlx") in ("llama-cpp", "omlx")
    if is_inkling_upstream:
        et_flag = bool(ct_kwargs["enable_thinking"]) if "enable_thinking" in ct_kwargs else None
        reasoning_effort = resolve_inkling_reasoning_effort(req_reasoning, enable_thinking=et_flag)
        enable_thinking = reasoning_effort != "none"
    else:
        if "enable_thinking" in ct_kwargs:
            enable_thinking = bool(ct_kwargs["enable_thinking"])
        elif str(model_name).endswith("-thinking"):
            enable_thinking = True
        elif req_reasoning and str(req_reasoning).lower() not in ("none", "off", "false", "minimal"):
            enable_thinking = True
        else:
            enable_thinking = False

        reasoning_effort = (
            str(req_reasoning).lower()
            if req_reasoning in ("low", "medium", "xhigh")
            else "medium"
        )

    default_temp = float(os.environ.get("M5_MLX_DEFAULT_TEMP", "0.6"))
    default_top_p = float(os.environ.get("M5_MLX_DEFAULT_TOP_P", "0.95"))
    default_top_k = int(os.environ.get("M5_MLX_DEFAULT_TOP_K", "0" if is_inkling_upstream else "20"))
    default_min_p = float(os.environ.get("M5_MLX_DEFAULT_MIN_P", "0.0"))
    default_presence_penalty = 0.0
    default_frequency_penalty = 0.0
    default_rep_penalty = 1.03
    temperature = float(body.get("temperature", default_temp))
    top_p = float(body.get("top_p", default_top_p))
    top_k = int(body.get("top_k", default_top_k))
    min_p = float(body.get("min_p", default_min_p))
    presence_penalty = float(body.get("presence_penalty", default_presence_penalty))
    frequency_penalty = float(body.get("frequency_penalty", default_frequency_penalty))
    repetition_penalty = float(body.get("repetition_penalty", default_rep_penalty))
    loop_info = detect_semantic_tool_loop(messages)
    temperature, repetition_penalty = resolve_sampling_with_loop_guard(
        messages, temperature, repetition_penalty
    )
    if loop_info.get("is_loop"):
        temperature = max(temperature, 0.9)
        presence_penalty = max(presence_penalty, 1.5)
        frequency_penalty = max(frequency_penalty, 0.4)
        repetition_penalty = max(repetition_penalty, 1.08)

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    loop = asyncio.get_running_loop()

    if not stream:
        try:
            result = await loop.run_in_executor(
                _MLX_EXECUTOR,
                lambda: ENGINE.generate_response(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                ),
            )
        except (RuntimeError, OSError) as exc:
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "code": "upstream_error",
                    }
                },
                status_code=502,
            )
        has_tools = bool(result["tool_calls"])
        finish_reason = "tool_calls" if has_tools else "stop"
        msg_obj: Dict[str, Any] = {
            "role": "assistant",
            "content": result["content"],
        }
        if result["reasoning_content"]:
            msg_obj["reasoning_content"] = result["reasoning_content"]
        if has_tools:
            msg_obj["tool_calls"] = result["tool_calls"]

        return JSONResponse(
            {
                "id": resp_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_obj,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                    "prompt_tokens_details": {"cached_tokens": result["cached_tokens"]},
                },
                "timings": {
                    "ttft_s": result["ttft_s"],
                    "decode_tok_s": result["decode_tok_s"],
                },
            }
        )

    async_queue: "asyncio.Queue[Any]" = asyncio.Queue()
    cancel_event = threading.Event()

    def _safe_put(val: Any) -> None:
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(async_queue.put_nowait, val)
        except RuntimeError:
            pass

    def _stream_worker():
        try:
            for item in ENGINE.generate_stream(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                cancel_event=cancel_event,
            ):
                _safe_put(item)
                if cancel_event.is_set():
                    break
        except Exception as exc:
            _safe_put(exc)
        finally:
            _safe_put(None)

    _MLX_EXECUTOR.submit(_stream_worker)

    async def event_stream():
        try:
            try:
                keepalive_s = float(os.environ.get("M5_MLX_SSE_KEEPALIVE_S", "5.0"))
            except ValueError:
                keepalive_s = 5.0
            role_chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"
            last_emit_ts = time.monotonic()

            final_result: Optional[Dict[str, Any]] = None
            stream_errored = False
            while True:
                try:
                    item = await asyncio.wait_for(async_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        cancel_event.set()
                        break
                    now = time.monotonic()
                    if keepalive_s > 0 and (now - last_emit_ts) >= keepalive_s:
                        keepalive_chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(keepalive_chunk, ensure_ascii=False)}\n\n"
                        last_emit_ts = now
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    stream_errored = True
                    err_chunk = {
                        "error": {
                            "message": str(item),
                            "type": "upstream_error",
                            "code": "upstream_error",
                        }
                    }
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                ev_type = item.get("type")
                if ev_type == "error":
                    stream_errored = True
                    err_chunk = {
                        "error": {
                            "message": str(item.get("message") or item.get("error") or "Upstream error"),
                            "type": "upstream_error",
                            "code": "upstream_error",
                        }
                    }
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                if ev_type == "reasoning":
                    r_txt = item.get("text") if "text" in item else item.get("delta", "")
                    r_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": r_txt},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(r_chunk, ensure_ascii=False)}\n\n"
                    last_emit_ts = time.monotonic()
                elif ev_type == "content":
                    c_txt = item.get("text") if "text" in item else item.get("delta", "")
                    c_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": c_txt},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n"
                    last_emit_ts = time.monotonic()
                elif ev_type == "final":
                    final_result = item["result"]

            if stream_errored or cancel_event.is_set():
                return

            if final_result is None:
                final_result = {
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": [],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "ttft_s": 0.0,
                    "decode_tok_s": 0.0,
                }

            has_tools = bool(final_result.get("tool_calls"))
            finish_reason = "tool_calls" if has_tools else "stop"

            if has_tools:
                tc_deltas = []
                for idx, tc in enumerate(final_result["tool_calls"]):
                    tc_deltas.append(
                        {
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                    )
                tc_chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

            done_chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": final_result["prompt_tokens"],
                    "completion_tokens": final_result["completion_tokens"],
                    "total_tokens": final_result["prompt_tokens"] + final_result["completion_tokens"],
                    "prompt_tokens_details": {"cached_tokens": final_result.get("cached_tokens", 0)},
                },
                "timings": {
                    "ttft_s": final_result.get("ttft_s", 0.0),
                    "decode_tok_s": final_result.get("decode_tok_s", 0.0),
                },
            }
            yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            cancel_event.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main(argv: Optional[List[str]] = None) -> None:
    global PACK_DIR, _SERVER_LOCK_FH

    m5_home = _get_m5_home()
    default_model_dir = _env_str(
        "M5_MLX_MODEL_DIR",
        str(m5_home / "models" / "Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"),
    )

    parser = argparse.ArgumentParser(description="Apple M5 MLX / llama.cpp / oMLX OpenAI Server")
    parser.add_argument(
        "--backend",
        choices=["omlx", "llama-cpp", "mlx"],
        default=_env_str("M5_MLX_BACKEND", "omlx"),
    )
    parser.add_argument("--host", default=_env_str("M5_MLX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("M5_MLX_PORT", 8080))
    parser.add_argument("--model-id", default=_env_str("M5_MLX_MODEL_ID", ""))
    parser.add_argument("--model-dir", default=default_model_dir)
    parser.add_argument("--context-length", type=int, default=_env_int("M5_MLX_CONTEXT_LENGTH", 131072))
    parser.add_argument("--kv-bits", default=_env_str("M5_MLX_KV_BITS", "4"))
    parser.add_argument("--kv-group-size", type=int, default=_env_int("M5_MLX_KV_GROUP_SIZE", 64))
    parser.add_argument("--quantized-kv-start", type=int, default=_env_int("M5_MLX_QUANTIZED_KV_START", 0))
    parser.add_argument("--prefill-step-size", type=int, default=_env_int("M5_MLX_PREFILL_STEP_SIZE", 256))
    parser.add_argument("--wired-limit-mb", type=int, default=_env_int("M5_MLX_WIRED_LIMIT_MB", 6144))
    parser.add_argument("--omlx-bin", default=_env_str("M5_MLX_OMLX_BIN", str(m5_home / "omlx-env" / "bin" / "omlx")))
    parser.add_argument("--omlx-model-dir", default=_env_str("M5_MLX_OMLX_MODEL_DIR", str(m5_home / "omlx-models")))
    parser.add_argument("--omlx-data-dir", default=_env_str("M5_MLX_OMLX_DATA_DIR", str(m5_home / "omlx-data")))
    parser.add_argument("--omlx-ssd-cache-dir", default=_env_str("M5_MLX_OMLX_SSD_CACHE_DIR", str(m5_home / "omlx-ssd-cache")))
    parser.add_argument("--omlx-port", type=int, default=_env_int("M5_MLX_OMLX_PORT", 8081))
    parser.add_argument("--llama-server-bin", default=_env_str("M5_MLX_LLAMA_SERVER_BIN", ""))
    parser.add_argument("--model-file", default=_env_str("M5_MLX_MODEL_FILE", ""))
    parser.add_argument("--draft-model-file", default=_env_str("M5_MLX_DRAFT_MODEL_FILE", ""))
    parser.add_argument("--spec-draft-max", type=int, default=_env_int("M5_MLX_SPEC_DRAFT_MAX", 0))
    parser.add_argument("--kv-cache-type", default=_env_str("M5_MLX_KV_CACHE_TYPE", "q4_0"))
    parser.add_argument("--llama-port", type=int, default=_env_int("M5_MLX_LLAMA_PORT", 8081))
    parser.add_argument("--threads", type=int, default=_env_int("M5_MLX_LLAMA_THREADS", 0))
    cli_args = parser.parse_args(argv)

    if not cli_args.model_id:
        if cli_args.backend == "omlx":
            cli_args.model_id = "inkling-2-mini-oq3e-mtp"
        elif cli_args.backend == "llama-cpp":
            cli_args.model_id = "inkling-2-mini-ud-q3-k-xl"
        else:
            cli_args.model_id = "qwen3.6-35b-a3b-3bit-mlx"

    try:
        validate_backend_config(
            backend=cli_args.backend,
            llama_server_bin=Path(cli_args.llama_server_bin) if cli_args.llama_server_bin else None,
            model_file=Path(cli_args.model_file) if cli_args.model_file else None,
            draft_model_file=Path(cli_args.draft_model_file) if cli_args.draft_model_file else None,
            llama_port=cli_args.llama_port,
            spec_draft_max=cli_args.spec_draft_max,
            kv_cache_type=cli_args.kv_cache_type,
            omlx_bin=Path(cli_args.omlx_bin) if cli_args.omlx_bin else None,
            omlx_model_dir=Path(cli_args.omlx_model_dir) if cli_args.omlx_model_dir else None,
            omlx_port=cli_args.omlx_port,
            model_id=cli_args.model_id,
            require_paths=(cli_args.backend in ("llama-cpp", "omlx")),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[m5-mlx-server] Invalid backend config: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)

    lock_name = "m5_mlx_server.lock" if int(cli_args.port) == 8080 else f"m5_mlx_server_{int(cli_args.port)}.lock"
    try:
        _SERVER_LOCK_FH = acquire_single_instance_lock(_get_m5_home() / lock_name)
    except RuntimeError as exc:
        print(f"[m5-mlx-server] Lock error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    PACK_DIR = Path(cli_args.model_dir)
    os.environ["M5_MLX_BACKEND"] = str(cli_args.backend)
    os.environ["M5_MLX_HOST"] = str(cli_args.host)
    os.environ["M5_MLX_PORT"] = str(cli_args.port)
    os.environ["M5_MLX_MODEL_DIR"] = str(PACK_DIR)
    os.environ["M5_MLX_MODEL_ID"] = str(cli_args.model_id)
    os.environ["M5_MLX_KV_BITS"] = str(cli_args.kv_bits)
    os.environ["M5_MLX_KV_GROUP_SIZE"] = str(cli_args.kv_group_size)
    os.environ["M5_MLX_QUANTIZED_KV_START"] = str(cli_args.quantized_kv_start)
    os.environ["M5_MLX_CONTEXT_LENGTH"] = str(cli_args.context_length)
    os.environ["M5_MLX_PREFILL_STEP_SIZE"] = str(cli_args.prefill_step_size)
    os.environ["M5_MLX_WIRED_LIMIT_MB"] = str(cli_args.wired_limit_mb)
    os.environ["M5_MLX_LLAMA_SERVER_BIN"] = str(cli_args.llama_server_bin)
    os.environ["M5_MLX_MODEL_FILE"] = str(cli_args.model_file)
    os.environ["M5_MLX_DRAFT_MODEL_FILE"] = str(cli_args.draft_model_file)
    os.environ["M5_MLX_SPEC_DRAFT_MAX"] = str(cli_args.spec_draft_max)
    os.environ["M5_MLX_KV_CACHE_TYPE"] = str(cli_args.kv_cache_type)
    os.environ["M5_MLX_LLAMA_PORT"] = str(cli_args.llama_port)
    os.environ["M5_MLX_OMLX_BIN"] = str(cli_args.omlx_bin)
    os.environ["M5_MLX_OMLX_MODEL_DIR"] = str(cli_args.omlx_model_dir)
    os.environ["M5_MLX_OMLX_DATA_DIR"] = str(cli_args.omlx_data_dir)
    os.environ["M5_MLX_OMLX_SSD_CACHE_DIR"] = str(cli_args.omlx_ssd_cache_dir)
    os.environ["M5_MLX_OMLX_PORT"] = str(cli_args.omlx_port)
    if cli_args.threads > 0:
        os.environ["M5_MLX_LLAMA_THREADS"] = str(cli_args.threads)

    atexit.register(_cleanup_on_exit)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, lambda signum, frame: (_cleanup_on_exit(), sys.exit(0)))
        except Exception:
            pass

    try:
        uvicorn.run(app, host=cli_args.host, port=cli_args.port, log_level="info")
    finally:
        _cleanup_on_exit()


if __name__ == "__main__":
    main()
