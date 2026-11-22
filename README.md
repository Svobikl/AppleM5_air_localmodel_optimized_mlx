# Apple M5 Air (24 GB) — Optimized Local MLX / oMLX OpenAI Server

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M5%20%2F%20M4%20%2F%20M3-000000?logo=apple)](https://www.apple.com/mac/)
[![MLX](https://img.shields.io/badge/Framework-MLX%20%2B%20oMLX%20%2B%20llama.cpp-1f6feb)](https://github.com/ml-explore/mlx)
[![Context](https://img.shields.io/badge/Context-128K%20(131%2C072%20tokens)-2ea043)](#why-this-exists--what-optimizations-are-included)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

A production-tuned, OpenAI-compatible (`/v1`) local LLM serving stack engineered specifically for the **Apple MacBook Air M5 with 24 GB Unified Memory** (and fully compatible with **24 GB+ M3 / M4 / M5** Apple Silicon Macs).

No model weights, virtual environments, or machine-specific paths are stored in Git. A single `./install.sh` command sets up isolated Python runtimes, downloads upstream Hugging Face checkpoints and Unsloth `llama.cpp` binaries, and wires the `m5-mlx-ctl` CLI.

---

## Hardware Target

- **Primary Target:** Apple MacBook Air M5, **24 GB Unified Memory** (macOS `Darwin arm64`)
- **Compatible Hardware:** Any Apple Silicon Mac (`M3`, `M4`, `M5`, Pro/Max/Ultra) with **24 GB or more** unified memory
- **16 GB Macs:** `./install.sh` will warn clearly; smaller GGUF/MLX profiles can still run, though full 128K context and 35B MoE models target 24 GB+ Macs.

---

## Why This Exists & What Optimizations Are Included

Running large sparse MoE models or 128K-context agentic workloads on a fanless 24 GB M5 MacBook Air hits three common failure modes in stock setups: **Metal GPU watchdog freezes (`409` / WindowServer stutter)**, **silent OOM crashes during long-context prefill**, and **macOS swap compression latency spikes** on idle MoE router weights. This repository addresses all three:

### 1. Hybrid 128K Context Routing for Inkling 2 Mini
- **Fast Path (`< 16,500` tokens):** Runs [`mlx-community/Inkling-2-Mini-oQ3e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp) on **oMLX v0.8.2** (`oQ3e` 3-bit quantization) at **~29–31 tok/s** with `--memory-guard-gb 20.0`, `--hot-cache-max-size 0`, `--hot-cache-write-through`, and `--max-concurrent-requests 1`.
- **Zero-OOM Sticky Overflow (`16,500`–`131,072` tokens):** Requests crossing the `16.5K` token boundary (or triggering an oMLX memory guard error before streaming begins) are serialized under an `RLock`: the server cleanly **stops oMLX first** to free Metal memory, then promotes to **Unsloth `llama-server`** running [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`Inkling-2-Mini-UD-Q3_K_XL.gguf`) with **`q4_0` KV cache** and **1024-token SWA** (`-c 131072 -ngl 99 -fa on -np 1 -b 512 -ub 64 --cache-reuse 256 -ctk q4_0 -ctv q4_0 -t 6`), staying under **~13.7–14.2 GB RSS** across the full 128K window.
- **Automatic Idle Restore:** Follow-up turns stay sticky on `llama-server` to avoid engine thrashing, then automatically restore oMLX after `>= 300s` idle on a short prompt (`< 4,000` tokens).

### 2. Qwen 3.6 35B-A3B 3-bit MoE Native MLX Engine
- **Mixed-Precision 3-bit MoE Layout:** Keeps routed MoE `switch_mlp` experts in **3-bit** (`< 4 GiB` total footprint), critical router gates (`mlp.gate`, `shared_expert_gate`) in **8-bit**, and attention / shared expert projections in **4-bit** (`group_size=64`). Can be pulled pre-quantized from [`unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit) or built locally via `scripts/build_qwen36_3bit_moe.py`.
- **4-bit KV Cache (`g64`):** Cuts attention cache footprint dramatically for long multi-turn agent histories.
- **Two-Tier Immutable `PrefixCacheManager`:** Separately caches system prompts and recent turn prefixes without LRU-evicting system prompts, cloning both standard `KVCache` layers and **GatedDeltaNet `ArraysCache`** state safely so subsequent requests never mutate cached prefixes.
- **Zero-Stutter GPU Frame Pacing:** `prefill_step_size=256` with a **`5 ms` GPU frame-slice yield** and `cancel_event` abort checks between prefill chunks so macOS `WindowServer` stays smooth and never hits watchdog `409` timeouts.
- **Bounded Metal & Swap Guardrails:** Sets `mx.set_cache_limit(128 MiB)`, clamps `mx.set_wired_limit` safely (`6144 MB`, clamped to `<= 50%` of Metal recommended max working set), and runs a lightweight **60s router keepwarm** pulse so macOS does not compress inactive MoE gate tensors into swap.

### 3. Production OpenAI-Compatible `/v1/chat/completions` Features
- **`5.0s` SSE Keepalive Chunks:** Emits periodic SSE keepalive comments during long prompt prefill so IDE clients and proxies never time out before the first token arrives.
- **Immediate Mid-Stream SSE Error Surfacing:** Upstream failures and prefill memory guards surface cleanly as OpenAI-formatted SSE error events instead of silent hangs or truncated streams.
- **14-Gram `<think>` Loop Guard:** Detects degenerate 14-gram repetition inside `<think>` reasoning blocks and breaks loops cleanly.
- **Multilingual-Safe Prose-Mask Anti-Loop:** Applies `presence_penalty`, `frequency_penalty`, and `repetition_penalty` strictly to multi-character prose tokens while protecting short words, digits, JSON punctuation (`{`, `}`), and XML/tool tags (`<tool_call>`, `<function=`, `<parameter=`).
- **XML & JSON Tool Calling:** Full parser support for both `<tool_call><function=...>...</function></tool_call>` XML payloads (with JSON-schema type coercion) and standard JSON tool calls.

---

## Supported Models & Upstream Sources

| Profile (`switch`) | Upstream Hugging Face Source | Quantization / Format | Active RAM / RSS | Context Window | Typical M5 Air Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `inkling2` *(default)* | [`mlx-community/Inkling-2-Mini-oQ3e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp) + [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | `oQ3e` 3-bit MLX (`<16.5K`) + `UD-Q3_K_XL` GGUF overflow (`16.5K..128K`) | ~10–14.2 GB RSS | `131,072` (128K) | ~29–31 tok/s (`oMLX`) / ~18–22 tok/s (`llama.cpp`) |
| `inkling2-gguf` | [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`Inkling-2-Mini-UD-Q3_K_XL.gguf`, optional `dspark-Inkling-2-Mini-Q8_0.gguf` draft) | `UD-Q3_K_XL` (`M5_MLX_SPEC_DRAFT_MAX=0` default on M5 Air; optional `Q8_0` draft), `q4_0` KV cache | ~13.7–14.2 GB RSS | `131,072` (128K) | ~18–24 tok/s |
| `qwen3.6` | [`unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit) *(or `--build-qwen-nvfp4` from [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) / [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4))* | 3-bit MoE experts / 8-bit gates / 4-bit attention + 4-bit KV (`g64`) | ~15–17 GB RSS | `131,072` (128K) | ~24–28 tok/s |

---

## One-Command Installation

```bash
git clone https://github.com/Svobikl/AppleM5_air_localmodel_optimized_mlx.git
cd AppleM5_air_localmodel_optimized_mlx

# Preview all planned virtualenvs, upstream URLs, and HF downloads without touching disk:
./install.sh --dry-run

# Full install (creates ~/.local/share/m5-mlx, installs mlx-env + omlx-env + Unsloth llama.cpp + all models):
./install.sh
```

### Installer Options

| Flag | Description |
| :--- | :--- |
| `--dry-run` | Print planned steps, upstream release URLs, HF repos, and target paths without writing anything (`exit 0`). |
| `--models <comma_list>` | Choose which models to download: `all` (default), `inkling2`, `inkling2-gguf`, `qwen3.6`. Example: `./install.sh --models inkling2,qwen3.6`. |
| `--skip-models` | Install Python environments (`mlx-env`, `omlx-env`), Unsloth `llama.cpp`, and CLI only; skip Hugging Face model downloads. |
| `--build-qwen-nvfp4` | Download `mlx-community/Qwen3.6-35B-A3B-4bit` (or supply `nvidia/Qwen3.6-35B-A3B-NVFP4`) and build the custom 3-bit MoE / 8-bit gate / 4-bit attention checkpoint via `scripts/build_qwen36_3bit_moe.py`. |
| `--enable-launchagent` | Install and load `~/Library/LaunchAgents/com.community.m5-mlx-server.plist` for automatic login startup. |
| `--home <dir>` | Override the data/runtime root (default: `$HOME/.local/share/m5-mlx`, or `M5_MLX_HOME`). |

> **Note:** Make sure `$HOME/.local/bin` is on your `PATH`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

---

## Usage & Model Switching

Manage the server lifecycle and active model profile with `m5-mlx-ctl`:

```bash
# Verify binaries, environments, weights, and system readiness
m5-mlx-ctl doctor

# Start the OpenAI-compatible server on http://127.0.0.1:8080/v1
m5-mlx-ctl start

# Check health, active backend, PID, and loaded model
m5-mlx-ctl status

# Tail server logs
m5-mlx-ctl logs

# Switch between supported model profiles (auto-restarts if running)
m5-mlx-ctl switch inkling2        # oMLX v0.8.2 + 128K GGUF overflow
m5-mlx-ctl switch inkling2-gguf   # Unsloth llama-server GGUF (M5_MLX_SPEC_DRAFT_MAX=0 default; optional Q8_0 draft)
m5-mlx-ctl switch qwen3.6         # NativeMLXEngine 3-bit MoE

# Stop the server cleanly
m5-mlx-ctl stop
```

Optional background service management:

```bash
m5-mlx-ctl install-launchagent
m5-mlx-ctl uninstall-launchagent
```

---

## Client Examples

All profiles expose an OpenAI-compatible API at **`http://127.0.0.1:8080/v1`**.

### `curl`

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool

curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini-oq3e-mtp",
    "messages": [{"role": "user", "content": "Explain MLX unified memory on Apple M5 in 3 bullets."}],
    "stream": true
  }'
```

### Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-m5-mlx",  # any non-empty string
)

resp = client.chat.completions.create(
    model="inkling-2-mini-oq3e-mtp",
    messages=[{"role": "user", "content": "Write a concise Python binary search."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

### Cursor / Continue / OpenWebUI

- **Base URL:** `http://127.0.0.1:8080/v1`
- **API Key:** `local-m5-mlx` (any non-empty placeholder)
- **Model ID:** check `curl http://127.0.0.1:8080/v1/models` (e.g., `inkling-2-mini-oq3e-mtp` or `qwen3.6-35b-a3b-3bit-mlx`)

---

## macOS Metal Memory Notes

On a **24 GB** Apple Silicon Mac, macOS defaults to `iogpu.wired_limit_mb=0`, which caps Metal's recommended maximum wired working set at roughly **~17.76 GiB** (`~74%` of physical RAM) to preserve headroom for macOS WindowServer and system processes.

- **Default Safe Behavior:** This server respects macOS defaults automatically (`M5_MLX_WIRED_LIMIT_MB=6144`, strictly clamped to `<= 50%` of Metal's `max_recommended_working_set_size`, plus `128 MiB` MLX cache limit) so you do **not** need `sudo` tweaks to run reliably.
- **Optional Tuning (Power Users):** If you dedicate the machine to large 128K-context runs and hit Metal allocation ceilings, you can temporarily raise the wired GPU limit until the next reboot:
  ```bash
  # Example: allow up to 19456 MB (19 GiB) wired GPU memory on a 24 GB Mac (resets on reboot)
  sudo sysctl iogpu.wired_limit_mb=19456
  ```
  Reset back to the macOS automatic default at any time with:
  ```bash
  sudo sysctl iogpu.wired_limit_mb=0
  ```

---

## Running the Test Suite

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## License

Licensed under the [Apache License 2.0](LICENSE).
