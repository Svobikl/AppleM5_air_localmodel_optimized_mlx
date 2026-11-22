#!/usr/bin/env python3
"""RED/GREEN unittest suite for Apple M5 Air 24GB MLX/oMLX community server (M1..M5)."""

import asyncio
import http.server
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_server():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("server.m5_mlx_server")


def _import_builder():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("scripts.build_qwen36_3bit_moe")


async def _call_asgi_get(asgi_app, path: str):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 18080),
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await asgi_app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(body.decode("utf-8"))


async def _call_asgi_post(asgi_app, path: str, payload: dict):
    raw_body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw_body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 18080),
    }
    sent = []
    consumed = False

    async def receive():
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)

    await asgi_app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(body.decode("utf-8"))


class _DummyKVCache:
    def __init__(self, shape=(1, 2, 4, 8)):
        self.keys = mx.zeros(shape, dtype=mx.float16)
        self.values = mx.zeros(shape, dtype=mx.float16)
        self.offset = shape[2]
        self.state = (self.keys, self.values)


class _DummyArraysCache:
    def __init__(self):
        self.cache = [mx.ones((1, 4), dtype=mx.float16), None]
        self.state = self.cache


class _DummyTokenizer:
    def __init__(self):
        self._vocab = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|endoftext|>": 3,
            "<tool_call>": 4,
            "</tool_call>": 5,
            "<function=": 6,
            "</function>": 7,
            "<parameter=": 8,
            "</parameter>": 9,
            "<think>": 10,
            "</think>": 11,
            "hello": 12,
            "world": 13,
            "the": 14,
            "1234": 15,
            "{": 16,
            "}": 17,
            "testing": 18,
        }
        self._id_to_tok = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def get_vocab_size(self):
        return len(self._vocab) + 1

    def token_to_id(self, tok: str):
        return self._vocab.get(tok)

    def decode(self, ids):
        return "".join(self._id_to_tok.get(int(i), "") for i in ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        tokens = []
        for idx, word in enumerate(text.split()):
            if word in self._vocab:
                tokens.append(self._vocab[word])
            else:
                tokens.append(12 + (idx % 2))
        return type("Enc", (), {"ids": tokens})()


class TestConfigLockAndAppM1(unittest.TestCase):
    """M1: M5_MLX_* env vars, exports, validate_backend_config, flock, wired cap, FastAPI app."""

    def test_expected_exports_and_env_vars_present(self):
        srv = _import_server()
        required_symbols = [
            "NativeMLXEngine",
            "OmlxBackendEngine",
            "LlamaCppBackendEngine",
            "PrefixCacheManager",
            "validate_backend_config",
            "compute_wired_limit_bytes",
            "collect_keepwarm_tensors",
            "normalize_messages_for_template",
            "_estimate_messages_tokens",
            "parse_model_output",
            "make_antiloop_logits_processor",
            "build_prose_mask",
            "has_repeated_ngram",
            "_owned_by_for_backend",
            "acquire_single_instance_lock",
            "app",
        ]
        for sym in required_symbols:
            self.assertTrue(hasattr(srv, sym), f"Missing export: {sym}")

        server_src = (REPO_ROOT / "server" / "m5_mlx_server.py").read_text(encoding="utf-8")
        expected_env_vars = [
            "M5_MLX_HOME",
            "M5_MLX_PORT",
            "M5_MLX_BACKEND",
            "M5_MLX_MODEL_ID",
            "M5_MLX_CONTEXT_LENGTH",
            "M5_MLX_WIRED_LIMIT_MB",
            "M5_MLX_PREFILL_STEP_SIZE",
            "M5_MLX_KV_BITS",
            "M5_MLX_KV_GROUP_SIZE",
            "M5_MLX_KEEPWARM_INTERVAL_S",
            "M5_MLX_SSE_KEEPALIVE_S",
            "M5_MLX_OMLX_BIN",
            "M5_MLX_OMLX_MODEL_DIR",
            "M5_MLX_OMLX_DATA_DIR",
            "M5_MLX_OMLX_SSD_CACHE_DIR",
            "M5_MLX_OMLX_PORT",
            "M5_MLX_OMLX_MEMORY_GUARD_GB",
            "M5_MLX_OMLX_MEMORY_GUARD",
            "M5_MLX_OMLX_HOT_CACHE",
            "M5_MLX_OMLX_MAX_CONCURRENT",
            "M5_MLX_OVERFLOW_TOKENS",
            "M5_MLX_RESTORE_IDLE_S",
            "M5_MLX_OVERFLOW_LLAMA_BIN",
            "M5_MLX_OVERFLOW_MODEL_FILE",
            "M5_MLX_LLAMA_SERVER_BIN",
            "M5_MLX_MODEL_FILE",
            "M5_MLX_LLAMA_PORT",
            "M5_MLX_KV_CACHE_TYPE",
            "M5_MLX_LLAMA_THREADS",
            "M5_MLX_SPEC_DRAFT_MAX",
            "M5_MLX_DRAFT_MODEL_FILE",
        ]
        for env_key in expected_env_vars:
            self.assertIn(env_key, server_src, f"Missing env var support: {env_key}")

    def test_validate_backend_config_valid_and_fail_closed(self):
        srv = _import_server()
        for backend in ("mlx", "omlx", "llama-cpp"):
            res = srv.validate_backend_config(backend=backend, llama_port=18081, omlx_port=18081)
            self.assertEqual(res["backend"], backend)

        with self.assertRaises(ValueError):
            srv.validate_backend_config(backend="invalid-backend")
        with self.assertRaises(ValueError):
            srv.validate_backend_config(backend="mlx", llama_port=0)
        with self.assertRaises(ValueError):
            srv.validate_backend_config(backend="omlx", omlx_port=70000)
        with self.assertRaises(ValueError):
            srv.validate_backend_config(backend="llama-cpp", spec_draft_max=-1)
        with self.assertRaises(ValueError):
            srv.validate_backend_config(backend="llama-cpp", kv_cache_type="q2_k")

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                srv.validate_backend_config(
                    backend="llama-cpp",
                    llama_server_bin=missing,
                    model_file=missing,
                    require_paths=True,
                )
            with self.assertRaises(FileNotFoundError):
                srv.validate_backend_config(
                    backend="omlx",
                    omlx_bin=missing,
                    omlx_model_dir=missing,
                    require_paths=True,
                )

    def test_acquire_single_instance_lock_flock_contention(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "m5_test.lock"
            fh = srv.acquire_single_instance_lock(lock_path)
            try:
                self.assertEqual(lock_path.read_text(encoding="utf-8").strip(), str(os.getpid()))
                with self.assertRaises(RuntimeError):
                    srv.acquire_single_instance_lock(lock_path)
            finally:
                fh.close()
            fh2 = srv.acquire_single_instance_lock(lock_path)
            fh2.close()

    def test_compute_wired_limit_bytes_clamps_to_half(self):
        srv = _import_server()
        rec_max = 16 * 1024**3
        clamped = srv.compute_wired_limit_bytes(12288, rec_max)
        self.assertEqual(clamped, int(0.5 * rec_max))
        self.assertLessEqual(clamped, rec_max // 2)

        bounded = srv.compute_wired_limit_bytes(4096, rec_max)
        self.assertEqual(bounded, 4096 * 1024 * 1024)
        self.assertEqual(srv.compute_wired_limit_bytes(0, rec_max), 0)
        self.assertEqual(srv.compute_wired_limit_bytes(4096, 0), 0)

    def test_collect_keepwarm_tensors(self):
        srv = _import_server()

        class _Mod:
            def __init__(self, shape):
                self.weight = mx.zeros(shape, dtype=mx.float16)

        class _Switch:
            def __init__(self):
                self.gate_proj = _Mod((4, 8))
                self.up_proj = _Mod((4, 8))
                self.down_proj = _Mod((4, 8))

        class _Mlp:
            def __init__(self):
                self.gate = _Mod((4, 8))
                self.shared_expert_gate = _Mod((1, 8))
                self.switch_mlp = _Switch()

        class _Layer:
            def __init__(self):
                self.mlp = _Mlp()

        class _Model:
            def __init__(self):
                self.layers = [_Layer(), _Layer()]

        tensors = srv.collect_keepwarm_tensors(_Model())
        self.assertEqual(len(tensors), 6)
        self.assertEqual(srv.collect_keepwarm_tensors(object()), [])

    def test_owned_by_and_fastapi_app_health_and_models(self):
        srv = _import_server()
        self.assertEqual(srv._owned_by_for_backend("omlx"), "m5-omlx")
        self.assertEqual(srv._owned_by_for_backend("llama-cpp"), "m5-llama-cpp")
        self.assertEqual(srv._owned_by_for_backend("mlx"), "m5-mlx")

        with mock.patch.object(srv, "ENGINE", None):
            with mock.patch.dict(os.environ, {"M5_MLX_BACKEND": "omlx"}, clear=False):
                st_h, health = asyncio.run(_call_asgi_get(srv.app, "/health"))
                self.assertEqual(st_h, 200)
                self.assertEqual(health["status"], "ok")
                self.assertEqual(health["backend"], "omlx")

                st_m, models = asyncio.run(_call_asgi_get(srv.app, "/v1/models"))
                self.assertEqual(st_m, 200)
                self.assertGreaterEqual(len(models["data"]), 1)
                self.assertEqual(models["data"][0]["owned_by"], "m5-omlx")


class TestMLXEngineAndParserM2(unittest.TestCase):
    """M2: PrefixCacheManager, ArraysCache immutability, cancel_event prefill abort, antiloop, XML/JSON tools."""

    def test_prefix_cache_manager_two_tier_and_arrayscache_immutability(self):
        srv = _import_server()
        mgr = srv.PrefixCacheManager(max_sys_entries=2, max_turn_entries=2)

        sys_tokens = list(range(100, 140))
        sys_cache = [_DummyKVCache(), _DummyArraysCache()]
        mgr.store(sys_tokens, sys_cache, is_sys=True)

        # Store 3 distinct turn entries so LRU evicts turn entries, never sys_entries.
        for base in (200, 300, 400):
            t_tokens = list(range(base, base + 40))
            mgr.store(t_tokens, [_DummyKVCache(), _DummyArraysCache()], is_sys=False)

        self.assertEqual(len(mgr.sys_entries), 1)
        self.assertEqual(len(mgr.turn_entries), 2)

        # Mutate returned ArraysCache and verify stored immutable prefix is preserved.
        matched_len, cloned_cache = mgr.find_longest_prefix(sys_tokens + [999], allow_turn_cache=False)
        self.assertEqual(matched_len, len(sys_tokens))
        self.assertIsNotNone(cloned_cache)
        cloned_cache[1].cache[0] = mx.zeros((1, 4), dtype=mx.float16)

        _, second_clone = mgr.find_longest_prefix(sys_tokens + [999], allow_turn_cache=False)
        self.assertEqual(float(mx.sum(second_clone[1].cache[0]).item()), 4.0)

        # Sequences shorter than 32 tokens must not be stored.
        mgr.store(list(range(1, 16)), [_DummyKVCache()], is_sys=True)
        self.assertEqual(len(mgr.sys_entries), 1)

    def test_native_mlx_engine_prefill_cancel_event_aborts(self):
        srv = _import_server()
        eng = srv.NativeMLXEngine.__new__(srv.NativeMLXEngine)
        eng.lock = threading.RLock()
        eng.prefix_cache = srv.PrefixCacheManager()
        eng._sys_len_cache = {}
        eng.kv_bits = None
        eng.kv_group_size = 64
        eng.quantized_kv_start = 0
        eng.context_length = 4096
        eng.prefill_step_size = 16
        eng.tokenizer = mock.MagicMock()
        # Return > 70 tokens so turn prefill attempts multiple steps.
        eng.tokenizer.encode.return_value = type("Enc", (), {"ids": list(range(100, 180))})()
        eng.template = mock.MagicMock()
        eng.template.render.return_value = "<|im_start|>user\n" + ("word " * 80) + "<|im_end|>"

        cancel_event = threading.Event()
        step_calls = []

        class _FakeModel:
            def make_cache(self):
                return [_DummyKVCache()]

            def model(self, sub, cache=None):
                del cache
                step_calls.append(int(sub.shape[-1]))
                cancel_event.set()
                return mx.zeros((1, int(sub.shape[-1]), 4), dtype=mx.float16)

        eng.model = _FakeModel()
        _, _, _, remaining_ids, _, _ = eng._prepare_prefill_cache(
            messages=[{"role": "user", "content": "word " * 80}],
            tools=None,
            max_tokens=64,
            enable_thinking=False,
            reasoning_effort="none",
            cancel_event=cancel_event,
        )
        self.assertEqual(remaining_ids, [])
        self.assertLessEqual(len(step_calls), 1)

    def test_has_repeated_ngram_14(self):
        srv = _import_server()
        gram = list(range(100, 114))
        self.assertTrue(srv.has_repeated_ngram(gram + [999] + gram, n=14))
        self.assertFalse(srv.has_repeated_ngram(list(range(100, 140)), n=14))
        self.assertFalse(srv.has_repeated_ngram(gram + gram[:-1], n=14))

    def test_build_prose_mask_and_antiloop_logits_processor(self):
        srv = _import_server()
        tok = _DummyTokenizer()
        protected = srv.build_protected_syntax_token_ids(tok)
        self.assertIn(tok.token_to_id("<tool_call>"), protected)
        self.assertIn(tok.token_to_id("{"), protected)

        vocab_size = tok.get_vocab_size()
        mask = srv.build_prose_mask(vocab_size, protected, tokenizer=tok)
        self.assertEqual(tuple(mask.shape), (vocab_size,))
        self.assertTrue(bool(mask[tok.token_to_id("hello")].item()))
        self.assertTrue(bool(mask[tok.token_to_id("testing")].item()))
        # Short word (< 4 chars), digits, and protected tags must be False.
        self.assertFalse(bool(mask[tok.token_to_id("the")].item()))
        self.assertFalse(bool(mask[tok.token_to_id("1234")].item()))
        self.assertFalse(bool(mask[tok.token_to_id("<tool_call>")].item()))

        proc = srv.make_antiloop_logits_processor(
            seed_token_ids=[tok.token_to_id("hello"), tok.token_to_id("<tool_call>")],
            protected_ids=protected,
            presence_penalty=1.2,
            frequency_penalty=0.25,
            repetition_penalty=1.05,
            prose_mask=mask,
        )
        logits = mx.zeros((1, vocab_size), dtype=mx.float32)
        out = proc(mx.array([tok.token_to_id("testing"), tok.token_to_id("{")]), logits)
        self.assertLess(float(out[0, tok.token_to_id("hello")].item()), 0.0)
        self.assertLess(float(out[0, tok.token_to_id("testing")].item()), 0.0)
        self.assertEqual(float(out[0, tok.token_to_id("<tool_call>")].item()), 0.0)
        self.assertEqual(float(out[0, tok.token_to_id("{")].item()), 0.0)

    def test_parse_model_output_xml_json_and_thinking(self):
        srv = _import_server()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_cmd",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "flag": {"type": "boolean"},
                            "opts": {"type": "object"},
                        },
                    },
                },
            }
        ]
        xml_raw = (
            "<think>planning step</think>\n"
            "Running now.\n"
            "<tool_call><function=run_cmd>"
            "<parameter=count>42</parameter>"
            "<parameter=flag>true</parameter>"
            '<parameter=opts>{"a": 1}</parameter>'
            "</function></tool_call>"
        )
        content, reasoning, tcs = srv.parse_model_output(xml_raw, tools=tools)
        self.assertEqual(content, "Running now.")
        self.assertEqual(reasoning, "planning step")
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "run_cmd")
        parsed_args = json.loads(tcs[0]["function"]["arguments"])
        self.assertEqual(parsed_args, {"count": 42, "flag": True, "opts": {"a": 1}})

        json_raw = '<tool_call>{"name": "search", "arguments": {"q": "mlx", "limit": 5}}</tool_call>'
        c2, r2, tcs2 = srv.parse_model_output(json_raw)
        self.assertEqual(c2, "")
        self.assertIsNone(r2)
        self.assertEqual(len(tcs2), 1)
        self.assertEqual(tcs2[0]["function"]["name"], "search")
        self.assertEqual(json.loads(tcs2[0]["function"]["arguments"]), {"q": "mlx", "limit": 5})

    def test_normalize_messages_and_estimate_tokens(self):
        srv = _import_server()
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Check status"},
            {
                "role": "assistant",
                "reasoning_content": "secret thought",
                "content": "<think>old inner monologue</think>\nCalling tool.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "status_tool", "arguments": '{"ok": true}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "healthy"},
            {"role": "system", "content": "Mid-turn reminder"},
        ]
        norm = srv.normalize_messages_for_template(msgs, strip_historical_thinking=True)
        self.assertEqual(norm[0]["role"], "system")
        self.assertEqual(norm[2]["reasoning_content"], "")
        self.assertEqual(norm[2]["content"], "Calling tool.")
        self.assertEqual(norm[3]["name"], "status_tool")
        self.assertEqual(norm[4]["role"], "user")
        self.assertIn("[System Note]", norm[4]["content"])

        self.assertEqual(srv._estimate_messages_tokens([], None), 0)
        est = srv._estimate_messages_tokens(norm, None)
        self.assertGreater(est, 0)


class TestOmlxAndOverflowM3(unittest.TestCase):
    """M3: OmlxBackendEngine, LlamaCppBackendEngine, RLock overflow/restore, negative no-retry, SSE error."""

    def test_build_omlx_server_cmd_defaults_and_tier_fallback(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            eng = srv.OmlxBackendEngine(
                omlx_bin=t / "omlx",
                omlx_model_dir=t / "models",
                omlx_data_dir=t / "data",
                omlx_ssd_cache_dir=t / "ssd",
                omlx_port=18081,
                start_subprocess=False,
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                cmd = eng._build_omlx_server_cmd()
            joined = " ".join(cmd)
            self.assertIn("--memory-guard-gb 20.0", joined)
            self.assertIn("--hot-cache-max-size 0", joined)
            self.assertIn("--hot-cache-write-through", joined)
            self.assertIn("--max-concurrent-requests 1", joined)

            with mock.patch.dict(
                os.environ,
                {
                    "M5_MLX_OMLX_MEMORY_GUARD": "tight",
                    "M5_MLX_OMLX_HOT_CACHE": "256",
                    "M5_MLX_OMLX_MAX_CONCURRENT": "2",
                },
                clear=True,
            ):
                cmd2 = eng._build_omlx_server_cmd()
            joined2 = " ".join(cmd2)
            self.assertIn("--memory-guard tight", joined2)
            self.assertNotIn("--memory-guard-gb", joined2)
            self.assertIn("--hot-cache-max-size 256", joined2)
            self.assertIn("--max-concurrent-requests 2", joined2)

    def test_write_model_settings_v2_atomic(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            data_dir = t / "data"
            data_dir.mkdir()
            (data_dir / "model_settings.json").write_text(
                json.dumps({"version": 1, "models": {"other-model": {"is_default": True}}}),
                encoding="utf-8",
            )
            eng = srv.OmlxBackendEngine(
                omlx_bin=t / "omlx",
                omlx_model_dir=t / "models",
                omlx_data_dir=data_dir,
                omlx_ssd_cache_dir=t / "ssd",
                omlx_port=18081,
                model_id="inkling-2-mini-oq3e-mtp",
                context_length=131072,
                start_subprocess=False,
            )
            settings_path = eng.write_model_settings()
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["version"], 2)
            self.assertFalse(loaded["models"]["other-model"]["is_default"])
            entry = loaded["models"]["inkling-2-mini-oq3e-mtp"]
            self.assertEqual(entry["max_context_window"], 131072)
            self.assertTrue(entry["is_pinned"])
            self.assertTrue(entry["is_default"])
            self.assertEqual(list(data_dir.glob("*.tmp")), [])

    def test_build_llama_server_cmd_flags(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            eng = srv.LlamaCppBackendEngine(
                llama_server_bin=t / "llama-server",
                model_file=t / "model.gguf",
                llama_port=18081,
                context_length=131072,
                kv_cache_type="q4_0",
                threads=6,
                spec_draft_max=0,
                start_subprocess=False,
            )
            cmd = eng._build_llama_server_cmd()
            joined = " ".join(cmd)
            for expected in (
                "-c 131072",
                "-ngl 99",
                "-fa on",
                "-np 1",
                "-b 512",
                "-ub 64",
                "--cache-reuse 256",
                "-ctk q4_0",
                "-ctv q4_0",
                "-t 6",
            ):
                self.assertIn(expected, joined)

    def test_serialized_proactive_overflow_sticky_reuse_and_idle_restore(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            llama_bin = t / "llama-server"
            llama_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            gguf = t / "model.gguf"
            gguf.write_text("gguf", encoding="utf-8")

            eng = srv.OmlxBackendEngine(
                omlx_bin=t / "omlx",
                omlx_model_dir=t / "models",
                omlx_data_dir=t / "data",
                omlx_ssd_cache_dir=t / "ssd",
                omlx_port=18081,
                overflow_llama_bin=llama_bin,
                overflow_model_file=gguf,
                overflow_tokens_threshold=16500,
                restore_idle_s=300.0,
                restore_max_tokens=4000,
                start_subprocess=False,
            )
            events = []

            def fake_omlx_shutdown():
                events.append("omlx_shutdown")

            def fake_omlx_start(startup_timeout_s=180.0):
                del startup_timeout_s
                events.append("omlx_start")

            class _StubOverflow:
                backend = "llama-cpp"

                def __init__(self, **kwargs):
                    del kwargs
                    events.append("llama_start")

                def shutdown(self):
                    events.append("llama_shutdown")

                def generate_stream(self, **kwargs):
                    del kwargs
                    yield {"type": "content", "delta": "from-llama"}
                    yield {"type": "final", "content": "from-llama", "reasoning": None, "tool_calls": []}

            eng.shutdown = fake_omlx_shutdown
            eng._start_omlx_server = fake_omlx_start
            eng.start_subprocess = True

            # Proactive overflow: >= 16500 estimated tokens -> must stop omlx BEFORE starting llama-server.
            large_msgs = [{"role": "user", "content": "x" * (17000 * 4)}]
            with (
                mock.patch.object(eng, "_wait_for_port_release"),
                mock.patch.object(srv, "LlamaCppBackendEngine", _StubOverflow),
            ):
                out1 = list(eng.generate_stream(large_msgs))
                self.assertEqual(out1[0]["delta"], "from-llama")
                self.assertEqual(events[:2], ["omlx_shutdown", "llama_start"])
                self.assertIsNotNone(eng._overflow_engine)

                # Short follow-up within < 300s idle must stay sticky on llama-cpp without restarting omlx.
                events.clear()
                short_msgs = [{"role": "user", "content": "hi"}]
                out2 = list(eng.generate_stream(short_msgs))
                self.assertEqual(out2[0]["delta"], "from-llama")
                self.assertEqual(events, [])

                # After >= 300s idle and < 4000 tokens, must restore omlx (stop llama BEFORE starting omlx).
                eng._last_activity_ts = time.monotonic() - 305.0
                with mock.patch.object(
                    eng,
                    "_generate_stream_unlocked",
                    return_value=iter([{"type": "content", "delta": "from-omlx"}]),
                ):
                    out3 = list(eng.generate_stream(short_msgs))
                self.assertEqual(out3[0]["delta"], "from-omlx")
                self.assertEqual(events, ["llama_shutdown", "omlx_start"])
                self.assertIsNone(eng._overflow_engine)

    def test_reactive_overflow_and_negative_no_retry_after_emit(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            llama_bin = t / "llama-server"
            llama_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            gguf = t / "model.gguf"
            gguf.write_text("gguf", encoding="utf-8")

            eng = srv.OmlxBackendEngine(
                omlx_bin=t / "omlx",
                omlx_model_dir=t / "models",
                omlx_data_dir=t / "data",
                omlx_ssd_cache_dir=t / "ssd",
                omlx_port=18081,
                overflow_llama_bin=llama_bin,
                overflow_model_file=gguf,
                start_subprocess=False,
            )

            class _StubOverflow:
                backend = "llama-cpp"

                def __init__(self, **kwargs):
                    del kwargs

                def shutdown(self):
                    pass

                def generate_stream(self, **kwargs):
                    del kwargs
                    yield {"type": "content", "delta": "recovered"}

            # 1) Reactive overflow succeeds when nothing was emitted yet.
            with (
                mock.patch.object(
                    eng,
                    "_generate_stream_unlocked",
                    side_effect=RuntimeError("omlx upstream SSE error: prefill memory guard triggered"),
                ),
                mock.patch.object(srv, "LlamaCppBackendEngine", _StubOverflow),
            ):
                evs = list(eng.generate_stream([{"role": "user", "content": "hi"}]))
            self.assertEqual(evs[0]["delta"], "recovered")

            # 2) Negative: once emitted_any is True, RuntimeError MUST NOT trigger overflow retry.
            eng._overflow_engine = None

            def _partial_then_fail(**kwargs):
                del kwargs
                yield {"type": "content", "delta": "partial"}
                raise RuntimeError("prefill memory guard after partial stream")

            with (
                mock.patch.object(eng, "_generate_stream_unlocked", side_effect=_partial_then_fail),
                mock.patch.object(eng, "_promote_to_overflow_locked") as promote_mock,
            ):
                gen = eng.generate_stream([{"role": "user", "content": "hi"}])
                first = next(gen)
                self.assertEqual(first["delta"], "partial")
                with self.assertRaises(RuntimeError):
                    next(gen)
                promote_mock.assert_not_called()

    def test_run_upstream_sse_pass_error_and_happy_path(self):
        srv = _import_server()
        mode = {"err": True}

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                del format, args

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if mode["err"]:
                    payload = json.dumps({"error": {"message": "prefill_memory_exceeded"}})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                else:
                    c1 = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
                    c2 = json.dumps(
                        {
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                        }
                    )
                    self.wfile.write(f"data: {c1}\n\n".encode("utf-8"))
                    self.wfile.write(f"data: {c2}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            eng = srv._BaseUpstreamHttpEngine()
            eng._init_upstream_base(
                upstream_host="127.0.0.1",
                upstream_port=port,
                context_length=4096,
                model_id="test-model",
                arch="test",
            )
            with self.assertRaises(RuntimeError) as ctx:
                list(
                    eng._run_upstream_sse_pass(
                        messages=[{"role": "user", "content": "hi"}],
                        tools=None,
                        max_tokens=32,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        min_p=0.0,
                        repetition_penalty=1.0,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        inkling_effort="none",
                        cancel_event=None,
                    )
                )
            self.assertIn("prefill_memory_exceeded", str(ctx.exception))

            mode["err"] = False
            events = list(
                eng._run_upstream_sse_pass(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                    max_tokens=32,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    inkling_effort="none",
                    cancel_event=None,
                )
            )
            self.assertEqual(events[0]["type"], "content")
            self.assertEqual(events[0]["delta"], "hello")
            self.assertEqual(events[-1]["type"], "pass_end")
            self.assertEqual(events[-1]["prompt_tokens"], 12)
        finally:
            server.shutdown()
            server.server_close()


class TestBuilderM4(unittest.TestCase):
    """M4: scripts/build_qwen36_3bit_moe.py synthetic quantization & missing source test."""

    def test_build_qwen36_3bit_moe_synthetic_shards(self):
        mod = _import_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            dst = root / "dst"
            src.mkdir()

            tensors = {
                "language_model.model.embed_tokens.weight": mx.zeros((128, 64), dtype=mx.float16),
                "language_model.model.layers.0.input_layernorm.weight": mx.ones((64,), dtype=mx.float16),
                "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64), dtype=mx.float16),
                "language_model.model.layers.0.self_attn.o_proj.weight": mx.zeros((64, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.gate.weight": mx.zeros((4, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.shared_expert_gate.weight": mx.zeros((1, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight": mx.zeros((64, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": mx.zeros((4, 64, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": mx.zeros((4, 64, 64), dtype=mx.float16),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": mx.zeros((4, 64, 64), dtype=mx.float16),
                "language_model.lm_head.weight": mx.zeros((128, 64), dtype=mx.float16),
            }
            shard_name = "model-00001-of-00001.safetensors"
            mx.save_safetensors(str(src / shard_name), tensors)
            (src / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": sum(v.nbytes for v in tensors.values())},
                        "weight_map": {k: shard_name for k in tensors},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (src / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {"model_type": "qwen3_5_moe", "num_hidden_layers": 1},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (src / "tokenizer.json").write_text("{}", encoding="utf-8")

            summary = mod.build_qwen36_3bit_moe(src, dst, group_size=64)
            self.assertIsInstance(summary, dict)
            out_cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
            qcfg = out_cfg.get("quantization") or out_cfg.get("text_config", {}).get("quantization") or {}

            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"language_model.model.layers.0.mlp.switch_mlp.{proj}"
                self.assertEqual(qcfg[key]["bits"], 3)
                self.assertEqual(qcfg[key]["group_size"], 64)

            self.assertEqual(qcfg["language_model.model.layers.0.mlp.gate"]["bits"], 8)
            self.assertEqual(qcfg["language_model.model.layers.0.mlp.gate"]["group_size"], 64)
            self.assertEqual(qcfg["language_model.model.layers.0.mlp.shared_expert_gate"]["bits"], 8)
            self.assertEqual(qcfg["language_model.model.layers.0.mlp.shared_expert_gate"]["group_size"], 64)

            self.assertEqual(qcfg["language_model.model.layers.0.self_attn.q_proj"]["bits"], 4)
            self.assertEqual(qcfg["language_model.model.layers.0.self_attn.q_proj"]["group_size"], 64)
            self.assertEqual(qcfg["language_model.model.layers.0.mlp.shared_expert.gate_proj"]["bits"], 4)
            self.assertEqual(qcfg["language_model.model.layers.0.mlp.shared_expert.gate_proj"]["group_size"], 64)

            out_tensors = mx.load(str(dst / shard_name))
            self.assertIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales", out_tensors)
            self.assertIn("language_model.model.layers.0.input_layernorm.weight", out_tensors)

    def test_build_qwen36_3bit_moe_missing_source_raises(self):
        mod = _import_builder()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mod.build_qwen36_3bit_moe(Path(tmp) / "does_not_exist", Path(tmp) / "dst")


class TestInstallerCtlPlistAndZeroTraceM4M5(unittest.TestCase):
    """M4..M5: install.sh --dry-run, m5-mlx-ctl bash -n & invalid switch, plist lint, 0-trace scan."""

    def test_install_sh_syntax_and_dry_run(self):
        install_sh = REPO_ROOT / "install.sh"
        self.assertTrue(install_sh.is_file(), f"Missing {install_sh}")
        syn = subprocess.run(["bash", "-n", str(install_sh)], capture_output=True, text=True)
        self.assertEqual(syn.returncode, 0, syn.stderr)

        dry = subprocess.run([str(install_sh), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        combined = dry.stdout + "\n" + dry.stderr
        for repo_id in (
            "mlx-community/Inkling-2-Mini-oQ3e-mtp",
            "unsloth/Inkling-2-Mini-GGUF",
            "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
        ):
            self.assertIn(repo_id, combined)

    def test_ctl_syntax_and_invalid_switch_exit_1(self):
        ctl = REPO_ROOT / "bin" / "m5-mlx-ctl"
        self.assertTrue(ctl.is_file(), f"Missing {ctl}")
        syn = subprocess.run(["bash", "-n", str(ctl)], capture_output=True, text=True)
        self.assertEqual(syn.returncode, 0, syn.stderr)

        bad = subprocess.run([str(ctl), "switch", "invalid_model"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)

    def test_launchd_plist_template_lint_and_no_hardcoded_home(self):
        plist = REPO_ROOT / "launchd" / "com.community.m5-mlx-server.plist.template"
        self.assertTrue(plist.is_file(), f"Missing {plist}")
        lint = subprocess.run(["plutil", "-lint", str(plist)], capture_output=True, text=True)
        self.assertEqual(lint.returncode, 0, lint.stderr)
        txt = plist.read_text(encoding="utf-8")
        self.assertIn("__M5_MLX_HOME__", txt)
        self.assertIn("com.community.m5-mlx-server", txt)

    def test_zero_forbidden_traces_across_repo(self):
        forbidden_ci = [
            "her" + "mes",
            "bonsai" + "2_server",
            "bonsai" + "2_openai_server",
            "macbook-air" + ".local",
        ]
        forbidden_exact = [
            "192." + "168.",
            "/" + "Users" + "/",
            "state" + ".db",
            "MEMORY" + ".md",
        ]
        skip_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}
        violations = []
        for path in REPO_ROOT.rglob("*"):
            if any(part in skip_dirs for part in path.parts) or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lowered = content.lower()
            rel = str(path.relative_to(REPO_ROOT))
            for term in forbidden_ci:
                if term in lowered:
                    violations.append(f"{rel}: contains {term!r}")
            for term in forbidden_exact:
                if term in content:
                    violations.append(f"{rel}: contains {term!r}")
        self.assertEqual(violations, [], "\n".join(violations))


class TestPatchRegressionsP1ToP7(unittest.TestCase):
    """RED->GREEN regression tests covering diff-review patches P1..P7."""

    @staticmethod
    def _create_synthetic_qwen_src(src_dir: Path) -> None:
        src_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            "language_model.model.embed_tokens.weight": mx.zeros((128, 64), dtype=mx.float16),
            "language_model.model.layers.0.input_layernorm.weight": mx.ones((64,), dtype=mx.float16),
            "language_model.model.layers.0.mlp.gate.weight": mx.zeros((4, 64), dtype=mx.float16),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": mx.zeros((4, 64, 64), dtype=mx.float16),
            "language_model.lm_head.weight": mx.zeros((128, 64), dtype=mx.float16),
        }
        shard_name = "model-00001-of-00001.safetensors"
        mx.save_safetensors(str(src_dir / shard_name), tensors)
        (src_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(v.nbytes for v in tensors.values())},
                    "weight_map": {k: shard_name for k in tensors},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (src_dir / "config.json").write_text(
            json.dumps(
                {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe", "num_hidden_layers": 1}},
                indent=2,
            ),
            encoding="utf-8",
        )
        (src_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    def test_p1_builder_cli_src_dst_and_source_output_match_install_sh(self):
        mod = _import_builder()
        install_txt = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn("--src", install_txt)
        self.assertIn("--dst", install_txt)
        self.assertNotIn("--source", install_txt)
        self.assertNotIn("--output", install_txt)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            dst1 = root / "dst1"
            dst2 = root / "dst2"
            self._create_synthetic_qwen_src(src)

            self.assertEqual(mod.main(["--src", str(src), "--dst", str(dst1)]), 0)
            self.assertTrue((dst1 / "config.json").is_file())
            self.assertEqual(mod.main(["--source", str(src), "--output", str(dst2)]), 0)
            self.assertTrue((dst2 / "config.json").is_file())

    def test_p2_overflow_llama_bin_and_gguf_paths_aligned(self):
        srv = _import_server()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            gguf = home / "models" / "Inkling-2-Mini-GGUF" / "Inkling-2-Mini-UD-Q3_K_XL.gguf"
            gguf.parent.mkdir(parents=True, exist_ok=True)
            gguf.write_text("gguf", encoding="utf-8")

            with mock.patch.dict(os.environ, {"M5_MLX_HOME": str(home)}, clear=False):
                os.environ.pop("M5_MLX_OVERFLOW_LLAMA_BIN", None)
                os.environ.pop("M5_MLX_OVERFLOW_MODEL_FILE", None)

                eng = srv.OmlxBackendEngine(
                    omlx_bin=home / "omlx",
                    omlx_model_dir=home / "omlx-models",
                    omlx_data_dir=home / "omlx-data",
                    omlx_ssd_cache_dir=home / "omlx-ssd-cache",
                    omlx_port=18081,
                    start_subprocess=False,
                )
                default_bin, default_gguf = eng._get_overflow_paths()
                self.assertEqual(default_bin, home / "bin" / "llama-server")
                self.assertEqual(default_gguf, gguf)
                self.assertFalse(eng._can_overflow())

                # Case A: canonical ${M5_MLX_HOME}/bin/llama-server exists
                canon_bin = home / "bin" / "llama-server"
                canon_bin.parent.mkdir(parents=True, exist_ok=True)
                canon_bin.write_text("#!/bin/sh\n", encoding="utf-8")
                canon_bin.chmod(0o755)
                resolved_bin, _ = eng._get_overflow_paths()
                self.assertEqual(resolved_bin, canon_bin)
                self.assertTrue(eng._can_overflow())

                # Case B: fallback ${M5_MLX_HOME}/bin/unsloth-llama-cpp/.../llama-server exists
                canon_bin.unlink()
                unsloth_bin = home / "bin" / "unsloth-llama-cpp" / "llama-b12588-mix-c0c3de8" / "llama-server"
                unsloth_bin.parent.mkdir(parents=True, exist_ok=True)
                unsloth_bin.write_text("#!/bin/sh\n", encoding="utf-8")
                unsloth_bin.chmod(0o755)
                fb_bin, _ = eng._get_overflow_paths()
                self.assertEqual(fb_bin, unsloth_bin)
                self.assertTrue(eng._can_overflow())

        plist_txt = (REPO_ROOT / "launchd" / "com.community.m5-mlx-server.plist.template").read_text(encoding="utf-8")
        self.assertIn("M5_MLX_OVERFLOW_LLAMA_BIN", plist_txt)
        self.assertIn("__M5_MLX_HOME__/bin/llama-server", plist_txt)
        self.assertIn("M5_MLX_OVERFLOW_MODEL_FILE", plist_txt)
        self.assertIn("Inkling-2-Mini-UD-Q3_K_XL.gguf", plist_txt)

    def test_p3_ctl_port_scoped_process_detection_and_plist_generation(self):
        ctl = REPO_ROOT / "bin" / "m5-mlx-ctl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "m5-home"
            fake_home = root / "user-home"
            (home / "mlx-env" / "bin").mkdir(parents=True, exist_ok=True)
            (home / "logs").mkdir(parents=True, exist_ok=True)
            (fake_home / "Library" / "LaunchAgents").mkdir(parents=True, exist_ok=True)

            # Stub python in ${M5_MLX_HOME}/mlx-env/bin/python that serves /health on --port
            stub_py = home / "mlx-env" / "bin" / "python"
            stub_py.write_text(
                f"""#!{sys.executable}
import http.server, json, sys, signal
port = 18080
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({{"status": "ok", "port": port}}).encode())
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
srv.serve_forever()
""",
                encoding="utf-8",
            )
            stub_py.chmod(0o755)

            # Launch an unrelated dummy process whose command line contains m5_mlx_server.py on port 19991
            dummy_proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)", "m5_mlx_server.py", "--port", "19991"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                env = os.environ.copy()
                env["HOME"] = str(fake_home)
                env["M5_MLX_HOME"] = str(home)
                env["M5_MLX_PORT"] = "19992"
                env["M5_MLX_UPSTREAM_PORT"] = "19993"

                sw = subprocess.run([str(ctl), "switch", "qwen3.6"], env=env, capture_output=True, text=True)
                self.assertEqual(sw.returncode, 0, sw.stderr)

                st = subprocess.run([str(ctl), "start"], env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(st.returncode, 0, st.stdout + "\n" + st.stderr)
                self.assertTrue((home / "m5-mlx-server-19992.pid").is_file())
                self.assertTrue((home / "logs" / "m5-mlx-server-19992.log").is_file())

                stat = subprocess.run([str(ctl), "status"], env=env, capture_output=True, text=True)
                self.assertEqual(stat.returncode, 0, stat.stdout + "\n" + stat.stderr)

                stop = subprocess.run([str(ctl), "stop"], env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(stop.returncode, 0, stop.stdout + "\n" + stop.stderr)
                # Ensure dummy process on 19991 was untouched
                self.assertIsNone(dummy_proc.poll())

                # Verify plist generation in isolated fake HOME
                inst_plist = subprocess.run(
                    [str(ctl), "install-launchagent"], env=env, capture_output=True, text=True
                )
                self.assertEqual(inst_plist.returncode, 0, inst_plist.stderr)
                plist_path = fake_home / "Library" / "LaunchAgents" / "com.community.m5-mlx-server.plist"
                self.assertTrue(plist_path.is_file())
                lint = subprocess.run(["plutil", "-lint", str(plist_path)], capture_output=True, text=True)
                self.assertEqual(lint.returncode, 0, lint.stderr)
                subprocess.run([str(ctl), "uninstall-launchagent"], env=env, capture_output=True, text=True)
            finally:
                subprocess.run([str(ctl), "stop"], env={**os.environ, "HOME": str(fake_home), "M5_MLX_HOME": str(home), "M5_MLX_PORT": "19992", "M5_MLX_UPSTREAM_PORT": "19993"}, capture_output=True, text=True)
                dummy_proc.terminate()
                dummy_proc.wait(timeout=5)

    def test_p4_setuptools_package_discovery_only_server(self):
        from setuptools.config.pyprojecttoml import apply_configuration
        from setuptools.dist import Distribution

        dist = Distribution()
        apply_configuration(dist, str(REPO_ROOT / "pyproject.toml"))
        self.assertEqual(dist.packages, ["server"])

    def test_p5_antiloop_processor_preserves_prose_mask_on_padded_vocab(self):
        srv = _import_server()
        tok = _DummyTokenizer()
        protected = srv.build_protected_syntax_token_ids(tok)
        vocab_size = tok.get_vocab_size()
        mask = srv.build_prose_mask(vocab_size, protected, tokenizer=tok)

        proc = srv.make_antiloop_logits_processor(
            seed_token_ids=[tok.token_to_id("hello"), tok.token_to_id("the"), tok.token_to_id("<tool_call>")],
            protected_ids=protected,
            presence_penalty=1.2,
            frequency_penalty=0.25,
            repetition_penalty=1.05,
            prose_mask=mask,
        )
        padded_vocab = vocab_size + 64
        logits = mx.zeros((1, padded_vocab), dtype=mx.float32)
        out = proc(mx.array([tok.token_to_id("testing"), tok.token_to_id("1234"), vocab_size + 5]), logits)

        # Prose words (>= 4 alpha) penalized
        self.assertLess(float(out[0, tok.token_to_id("hello")].item()), 0.0)
        self.assertLess(float(out[0, tok.token_to_id("testing")].item()), 0.0)
        # Short word, digits, protected syntax, and padded out-of-vocab IDs stay strictly 0.0
        self.assertEqual(float(out[0, tok.token_to_id("the")].item()), 0.0)
        self.assertEqual(float(out[0, tok.token_to_id("1234")].item()), 0.0)
        self.assertEqual(float(out[0, tok.token_to_id("<tool_call>")].item()), 0.0)
        self.assertEqual(float(out[0, vocab_size + 5].item()), 0.0)

    def test_p6_case_check_draft_persistence_and_model_id_alignment(self):
        srv = _import_server()
        ctl = REPO_ROOT / "bin" / "m5-mlx-ctl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "m5-home"
            fake_home = root / "user-home"
            (home / "omlx-models" / "inkling-2-mini-oq3e-mtp").mkdir(parents=True, exist_ok=True)
            (fake_home / "Library" / "LaunchAgents").mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "HOME": str(fake_home), "M5_MLX_HOME": str(home)}

            doc = subprocess.run([str(ctl), "doctor"], env=env, capture_output=True, text=True)
            self.assertEqual(doc.returncode, 0, doc.stderr)
            self.assertIn("[OK]   Model present:", doc.stdout)
            self.assertIn("omlx-models/inkling-2-mini-oq3e-mtp", doc.stdout)
            self.assertNotIn("omlx-models/Inkling-2-Mini-oQ3e-mtp", doc.stdout)

            # Switch to inkling2-gguf: persists M5_MLX_DRAFT_MODEL_FILE & M5_MLX_SPEC_DRAFT_MAX=0
            sw1 = subprocess.run([str(ctl), "switch", "inkling2-gguf"], env=env, capture_output=True, text=True)
            self.assertEqual(sw1.returncode, 0, sw1.stderr)
            prof1 = (home / "active_profile.env").read_text(encoding="utf-8")
            self.assertIn("M5_MLX_DRAFT_MODEL_FILE=", prof1)
            self.assertIn("M5_MLX_SPEC_DRAFT_MAX=0", prof1)

            # Switch back to inkling2: clears M5_MLX_DRAFT_MODEL_FILE
            sw2 = subprocess.run([str(ctl), "switch", "inkling2"], env=env, capture_output=True, text=True)
            self.assertEqual(sw2.returncode, 0, sw2.stderr)
            prof2 = (home / "active_profile.env").read_text(encoding="utf-8")
            self.assertNotIn("M5_MLX_DRAFT_MODEL_FILE=", prof2)

            # Switch to qwen3.6: model_id must match canonical qwen3.6-35b-a3b-3bit-mlx
            sw3 = subprocess.run([str(ctl), "switch", "qwen3.6"], env=env, capture_output=True, text=True)
            self.assertEqual(sw3.returncode, 0, sw3.stderr)
            prof3 = (home / "active_profile.env").read_text(encoding="utf-8")
            self.assertIn("M5_MLX_MODEL_ID=qwen3.6-35b-a3b-3bit-mlx", prof3)

        readme_txt = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("qwen3.6-35b-a3b-3bit-mlx", readme_txt)
        self.assertNotIn("qwen3.6-35b-a3b-3bit-moe", readme_txt)
        self.assertIn("M5_MLX_SPEC_DRAFT_MAX=0", readme_txt)

        install_txt = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn('M5_MLX_SPEC_DRAFT_MAX="0"', install_txt)

        with mock.patch.object(srv, "ENGINE", None):
            with mock.patch.dict(os.environ, {"M5_MLX_BACKEND": "mlx"}, clear=False):
                os.environ.pop("M5_MLX_MODEL_ID", None)
                _, health = asyncio.run(_call_asgi_get(srv.app, "/health"))
                self.assertEqual(health["model"], "qwen3.6-35b-a3b-3bit-mlx")
                _, models = asyncio.run(_call_asgi_get(srv.app, "/v1/models"))
                self.assertEqual(models["data"][0]["id"], "qwen3.6-35b-a3b-3bit-mlx")
                _, tags = asyncio.run(_call_asgi_get(srv.app, "/api/tags"))
                tag_names = [m["name"] for m in tags["models"]]
                self.assertIn("qwen3.6-35b-a3b-3bit-mlx", tag_names)

    def test_p7_non_stream_chat_completions_oserror_returns_502(self):
        srv = _import_server()

        class _FailEngine:
            backend = "omlx"

            def generate_response(self, **kwargs):
                del kwargs
                raise OSError("upstream socket reset")

        with mock.patch.object(srv, "ENGINE", _FailEngine()):
            st, data = asyncio.run(
                _call_asgi_post(
                    srv.app,
                    "/v1/chat/completions",
                    {"model": "inkling-2-mini-oq3e-mtp", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                )
            )
        self.assertEqual(st, 502)
        self.assertEqual(data["error"]["type"], "upstream_error")
        self.assertIn("upstream socket reset", data["error"]["message"])

        class _OkEngine:
            backend = "omlx"

            def generate_response(self, **kwargs):
                del kwargs
                return {
                    "content": "ok-reply",
                    "reasoning_content": None,
                    "tool_calls": [],
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "ttft_s": 0.01,
                    "decode_tok_s": 10.0,
                }

        with mock.patch.object(srv, "ENGINE", _OkEngine()):
            st_ok, data_ok = asyncio.run(
                _call_asgi_post(
                    srv.app,
                    "/v1/chat/completions",
                    {"model": "inkling-2-mini-oq3e-mtp", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                )
            )
        self.assertEqual(st_ok, 200)
        self.assertEqual(data_ok["choices"][0]["message"]["content"], "ok-reply")

        # Empty-string env vars (e.g., unset profile fields) must not crash main() argparse
        with (
            mock.patch.dict(
                os.environ,
                {"M5_MLX_KV_GROUP_SIZE": "", "M5_MLX_SPEC_DRAFT_MAX": "", "M5_MLX_PORT": ""},
                clear=False,
            ),
            mock.patch.object(srv.uvicorn, "run") as uv_run,
            mock.patch.object(srv, "acquire_single_instance_lock", return_value=mock.MagicMock()),
        ):
            srv.main(["--backend", "mlx", "--port", "18999"])
            uv_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
