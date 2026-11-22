#!/usr/bin/env bash
# Community installer for Apple M5 MacBook Air (24 GB) MLX / oMLX / llama.cpp OpenAI server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M5_MLX_HOME="${M5_MLX_HOME:-$HOME/.local/share/m5-mlx}"
DRY_RUN=0
SKIP_MODELS=0
MODELS_ARG="all"
BUILD_QWEN_NVFP4=0
ENABLE_LAUNCHAGENT=0

OMLX_VERSION="0.8.2"
UNSLOTH_LLAMA_TAG="b12588-mix-c0c3de8"
UNSLOTH_LLAMA_URL="https://github.com/unslothai/llama.cpp/releases/download/${UNSLOTH_LLAMA_TAG}/llama-${UNSLOTH_LLAMA_TAG}-bin-macos-arm64.tar.gz"

HF_INKLING_OMLX="mlx-community/Inkling-2-Mini-oQ3e-mtp"
HF_INKLING_GGUF="unsloth/Inkling-2-Mini-GGUF"
HF_QWEN_3BIT="unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit"
HF_QWEN_4BIT="mlx-community/Qwen3.6-35B-A3B-4bit"
HF_QWEN_NVFP4="nvidia/Qwen3.6-35B-A3B-NVFP4"

usage() {
  cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Install virtual environments, upstream binaries, Hugging Face models, and CLI
tools for the Apple M5 Air 24 GB MLX / oMLX / llama.cpp OpenAI server.

Options:
  --dry-run              Print planned actions, URLs, and paths without writing files
  --skip-models          Install runtimes and CLI only; skip Hugging Face downloads
  --models <comma_list>  Models to download: all, inkling2, inkling2-gguf, qwen3.6
                         (default: all)
  --build-qwen-nvfp4     Build custom 3-bit MoE / 8-bit gate / 4-bit attn Qwen3.6
                         artifact via scripts/build_qwen36_3bit_moe.py instead of
                         downloading the pre-quantized 3-bit MLX checkpoint
  --enable-launchagent   Install and load ~/Library/LaunchAgents/com.community.m5-mlx-server.plist
  --home <dir>           Override installation directory (default: $HOME/.local/share/m5-mlx)
  -h, --help             Show this help message and exit
EOF
}

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_err() {
  printf '[ERROR] %s\n' "$*" >&2
}

run_or_dry() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] %s\n' "$*"
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --models)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        log_err "--models requires a comma-separated value (all, inkling2, inkling2-gguf, qwen3.6)"
        exit 1
      fi
      MODELS_ARG="$2"
      shift 2
      ;;
    --build-qwen-nvfp4)
      BUILD_QWEN_NVFP4=1
      shift
      ;;
    --enable-launchagent)
      ENABLE_LAUNCHAGENT=1
      shift
      ;;
    --home)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        log_err "--home requires a directory path"
        exit 1
      fi
      M5_MLX_HOME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_err "Unknown option: $1"
      usage >&2
      exit 1
      ;;
  esac
done

# Parse and validate --models list
WANT_INKLING2=0
WANT_INKLING2_GGUF=0
WANT_QWEN36=0

IFS=',' read -r -a _raw_models <<< "$MODELS_ARG"
for _m in "${_raw_models[@]}"; do
  _trimmed="${_m//[[:space:]]/}"
  case "$_trimmed" in
    all)
      WANT_INKLING2=1
      WANT_INKLING2_GGUF=1
      WANT_QWEN36=1
      ;;
    inkling2|inkling2-omlx)
      WANT_INKLING2=1
      ;;
    inkling2-gguf)
      WANT_INKLING2_GGUF=1
      ;;
    qwen3.6)
      WANT_QWEN36=1
      ;;
    "")
      ;;
    *)
      log_err "Unknown model selector in --models: '$_trimmed' (expected: all, inkling2, inkling2-gguf, qwen3.6)"
      exit 1
      ;;
  esac
done

if [[ "$WANT_INKLING2" -eq 0 && "$WANT_INKLING2_GGUF" -eq 0 && "$WANT_QWEN36" -eq 0 ]]; then
  log_err "No valid models selected via --models='$MODELS_ARG'"
  exit 1
fi

# Prerequisite checks: macOS + Apple Silicon (Darwin arm64)
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [[ "$OS_NAME" != "Darwin" || "$ARCH_NAME" != "arm64" ]]; then
  log_err "Unsupported platform: ${OS_NAME} ${ARCH_NAME}. Apple Silicon macOS (Darwin arm64) is required."
  exit 1
fi

MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
log_info "Detected macOS Apple Silicon (${OS_NAME} ${ARCH_NAME}), ${MEM_GB} GB unified memory."
if [[ "$MEM_GB" -lt 24 ]]; then
  log_warn "Detected ${MEM_GB} GB unified memory (< 24 GB). Large 3-bit MoE and 128K context profiles are tuned for 24 GB+ Macs and may swap or hit Metal memory limits."
fi

# Detect uv or fallback to python3 -m venv + pip
USE_UV=0
if command -v uv >/dev/null 2>&1; then
  USE_UV=1
  log_info "Using uv ($(command -v uv)) for environment setup."
else
  log_warn "uv not found; falling back to python3 -m venv + pip (install uv via 'brew install uv' for faster setup)."
fi

select_python_for_omlx() {
  local candidate ver tag
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
      case "$ver" in
        3.13) tag="313"; printf '%s %s %s\n' "$candidate" "$ver" "$tag"; return 0 ;;
        3.12) tag="312"; printf '%s %s %s\n' "$candidate" "$ver" "$tag"; return 0 ;;
        3.11) tag="311"; printf '%s %s %s\n' "$candidate" "$ver" "$tag"; return 0 ;;
      esac
    fi
  done
  return 1
}

OMLX_PY_BIN="python3"
OMLX_PY_VER="3.13"
OMLX_PY_TAG="313"

if [[ "$USE_UV" -eq 1 ]]; then
  if _found="$(select_python_for_omlx 2>/dev/null)"; then
    read -r OMLX_PY_BIN OMLX_PY_VER OMLX_PY_TAG <<< "$_found"
  fi
  # Prefer 3.13 with uv (uv manages standalone CPython builds if needed)
  OMLX_PY_VER="3.13"
  OMLX_PY_TAG="313"
else
  if ! _found="$(select_python_for_omlx 2>/dev/null)"; then
    log_err "oMLX v${OMLX_VERSION} requires Python 3.11, 3.12, or 3.13 (or install uv)."
    exit 1
  fi
  read -r OMLX_PY_BIN OMLX_PY_VER OMLX_PY_TAG <<< "$_found"
fi

OMLX_WHEEL_URL="https://github.com/jundot/omlx/releases/download/v${OMLX_VERSION}/omlx-${OMLX_VERSION}-cp${OMLX_PY_TAG}-cp${OMLX_PY_TAG}-macosx_15_0_universal2.whl"

ENVS_DIR="${M5_MLX_HOME}/envs"
MLX_VENV="${ENVS_DIR}/mlx-env"
OMLX_VENV="${ENVS_DIR}/omlx-env"
LLAMA_DEST="${M5_MLX_HOME}/bin/unsloth-llama-cpp"
MODELS_DIR="${M5_MLX_HOME}/models"
OMLX_MODELS_DIR="${M5_MLX_HOME}/omlx-models"
OMLX_DATA_DIR="${M5_MLX_HOME}/omlx-data"
OMLX_SSD_DIR="${M5_MLX_HOME}/omlx-ssd-cache"
LOCAL_BIN_DIR="$HOME/.local/bin"
ACTIVE_PROFILE="${M5_MLX_HOME}/active_profile.env"

log_info "Target M5_MLX_HOME: ${M5_MLX_HOME}"
log_info "Repo directory:     ${SCRIPT_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log_info "Dry-run mode enabled: no files, virtualenvs, symlinks, or downloads will be written."
  printf '\nPlanned installation steps:\n'
  printf '  1. Create directories under %s/{envs,bin,logs,models,omlx-models,omlx-data,omlx-ssd-cache} and %s\n' "$M5_MLX_HOME" "$LOCAL_BIN_DIR"
  if [[ "$USE_UV" -eq 1 ]]; then
    printf '  2. Create MLX venv via uv at %s (plus compatibility symlink %s/mlx-env) and install %s/requirements.txt\n' "$MLX_VENV" "$M5_MLX_HOME" "$SCRIPT_DIR"
    printf '  3. Create oMLX venv via uv (--python %s) at %s (plus compatibility symlink %s/omlx-env) and install %s\n' "$OMLX_PY_VER" "$OMLX_VENV" "$M5_MLX_HOME" "$OMLX_WHEEL_URL"
  else
    printf '  2. Create MLX venv via python3 -m venv at %s (plus compatibility symlink %s/mlx-env) and pip install -r %s/requirements.txt\n' "$MLX_VENV" "$M5_MLX_HOME" "$SCRIPT_DIR"
    printf '  3. Create oMLX venv via %s -m venv at %s (plus compatibility symlink %s/omlx-env) and pip install %s\n' "$OMLX_PY_BIN" "$OMLX_VENV" "$M5_MLX_HOME" "$OMLX_WHEEL_URL"
  fi
  printf '  4. Download & extract Unsloth llama.cpp (%s) from:\n       %s\n     into %s (and symlink %s/bin/llama-server)\n' "$UNSLOTH_LLAMA_TAG" "$UNSLOTH_LLAMA_URL" "$LLAMA_DEST" "$M5_MLX_HOME"
  if [[ "$SKIP_MODELS" -eq 1 ]]; then
    printf '  5. Skip Hugging Face model downloads (--skip-models)\n'
  else
    printf '  5. Download selected Hugging Face models via huggingface_hub.snapshot_download:\n'
    if [[ "$WANT_INKLING2" -eq 1 ]]; then
      printf '       - %s -> %s/Inkling-2-Mini-oQ3e-mtp (symlinked at %s/inkling-2-mini-oq3e-mtp)\n' "$HF_INKLING_OMLX" "$MODELS_DIR" "$OMLX_MODELS_DIR"
    fi
    if [[ "$WANT_INKLING2_GGUF" -eq 1 ]]; then
      printf '       - %s (Inkling-2-Mini-UD-Q3_K_XL.gguf, dspark-Inkling-2-Mini-Q8_0.gguf) -> %s/Inkling-2-Mini-GGUF/\n' "$HF_INKLING_GGUF" "$MODELS_DIR"
    fi
    if [[ "$WANT_QWEN36" -eq 1 ]]; then
      if [[ "$BUILD_QWEN_NVFP4" -eq 1 ]]; then
        printf '       - %s (or %s) -> build via %s/scripts/build_qwen36_3bit_moe.py -> %s/Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE\n' "$HF_QWEN_4BIT" "$HF_QWEN_NVFP4" "$SCRIPT_DIR" "$MODELS_DIR"
      else
        printf '       - %s -> %s/Qwen3.6-35B-A3B-UD-MLX-3bit\n' "$HF_QWEN_3BIT" "$MODELS_DIR"
      fi
    fi
  fi
  printf '  6. Write default profile %s and symlink %s/bin/m5-mlx-ctl -> %s/m5-mlx-ctl\n' "$ACTIVE_PROFILE" "$SCRIPT_DIR" "$LOCAL_BIN_DIR"
  if [[ "$ENABLE_LAUNCHAGENT" -eq 1 ]]; then
    printf '  7. Install & load LaunchAgent from %s/launchd/com.community.m5-mlx-server.plist.template\n' "$SCRIPT_DIR"
  else
    printf '  7. LaunchAgent disabled by default (pass --enable-launchagent or run m5-mlx-ctl install-launchagent later)\n'
  fi
  exit 0
fi

# Step 1: Create directory structure
mkdir -p \
  "$ENVS_DIR" \
  "${M5_MLX_HOME}/bin" \
  "$LLAMA_DEST" \
  "${M5_MLX_HOME}/logs" \
  "$MODELS_DIR" \
  "$OMLX_MODELS_DIR" \
  "$OMLX_DATA_DIR" \
  "$OMLX_SSD_DIR" \
  "$LOCAL_BIN_DIR"

# Step 2: Create mlx-env and install requirements.txt
log_info "Setting up mlx-env at ${MLX_VENV}..."
if [[ "$USE_UV" -eq 1 ]]; then
  uv venv --allow-existing "$MLX_VENV"
  uv pip install --python "${MLX_VENV}/bin/python" -r "${SCRIPT_DIR}/requirements.txt"
else
  if [[ ! -x "${MLX_VENV}/bin/python" ]]; then
    python3 -m venv "$MLX_VENV"
  fi
  "${MLX_VENV}/bin/python" -m pip install --upgrade pip
  "${MLX_VENV}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi
ln -sfn "$MLX_VENV" "${M5_MLX_HOME}/mlx-env"

# Step 3: Create omlx-env and install oMLX v0.8.2 wheel
log_info "Setting up omlx-env (Python ${OMLX_PY_VER}, cp${OMLX_PY_TAG}) at ${OMLX_VENV}..."
if [[ "$USE_UV" -eq 1 ]]; then
  uv venv --allow-existing --python "$OMLX_PY_VER" "$OMLX_VENV"
  uv pip install --python "${OMLX_VENV}/bin/python" "$OMLX_WHEEL_URL"
else
  if [[ ! -x "${OMLX_VENV}/bin/python" ]]; then
    "$OMLX_PY_BIN" -m venv "$OMLX_VENV"
  fi
  "${OMLX_VENV}/bin/python" -m pip install --upgrade pip
  "${OMLX_VENV}/bin/python" -m pip install "$OMLX_WHEEL_URL"
fi
ln -sfn "$OMLX_VENV" "${M5_MLX_HOME}/omlx-env"

# Step 4: Download & extract Unsloth llama.cpp macOS arm64 binary
log_info "Downloading Unsloth llama.cpp (${UNSLOTH_LLAMA_TAG}) into ${LLAMA_DEST}..."
TMP_TGZ="$(mktemp "${TMPDIR:-/tmp}/unsloth-llama-XXXXXX.tar.gz")"
trap 'rm -f "$TMP_TGZ"' EXIT
curl -fL --retry 3 --retry-delay 2 -o "$TMP_TGZ" "$UNSLOTH_LLAMA_URL"
tar -xzf "$TMP_TGZ" -C "$LLAMA_DEST"
rm -f "$TMP_TGZ"
trap - EXIT

LLAMA_SERVER_FOUND="$(find "$LLAMA_DEST" -maxdepth 4 -type f -name 'llama-server' | head -n 1 || true)"
if [[ -n "$LLAMA_SERVER_FOUND" ]]; then
  chmod +x "$LLAMA_SERVER_FOUND"
  ln -sfn "$LLAMA_SERVER_FOUND" "${M5_MLX_HOME}/bin/llama-server"
  log_info "Linked llama-server -> ${LLAMA_SERVER_FOUND}"
else
  log_warn "llama-server binary not found inside ${LLAMA_DEST}; verify archive layout."
fi

# Step 5: Download Hugging Face models via huggingface_hub
hf_snapshot_download() {
  local repo_id="$1"
  local local_dir="$2"
  local patterns_csv="${3:-}"
  log_info "Downloading ${repo_id} -> ${local_dir}..."
  "${MLX_VENV}/bin/python" - "$repo_id" "$local_dir" "$patterns_csv" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, patterns_csv = sys.argv[1], sys.argv[2], sys.argv[3]
allow_patterns = [p for p in patterns_csv.split(",") if p] if patterns_csv else None
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)
PYEOF
}

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  if [[ "$WANT_INKLING2" -eq 1 ]]; then
    INKLING_DIR="${MODELS_DIR}/Inkling-2-Mini-oQ3e-mtp"
    hf_snapshot_download "$HF_INKLING_OMLX" "$INKLING_DIR"
    ln -sfn "$INKLING_DIR" "${OMLX_MODELS_DIR}/inkling-2-mini-oq3e-mtp"
  fi

  if [[ "$WANT_INKLING2_GGUF" -eq 1 ]]; then
    INKLING_GGUF_DIR="${MODELS_DIR}/Inkling-2-Mini-GGUF"
    hf_snapshot_download \
      "$HF_INKLING_GGUF" \
      "$INKLING_GGUF_DIR" \
      "Inkling-2-Mini-UD-Q3_K_XL.gguf,dspark-Inkling-2-Mini-Q8_0.gguf"
  fi

  if [[ "$WANT_QWEN36" -eq 1 ]]; then
    if [[ "$BUILD_QWEN_NVFP4" -eq 1 ]]; then
      QWEN_SRC_DIR="${MODELS_DIR}/Qwen3.6-35B-A3B-4bit"
      QWEN_OUT_DIR="${MODELS_DIR}/Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"
      hf_snapshot_download "$HF_QWEN_4BIT" "$QWEN_SRC_DIR"
      log_info "Building 3-bit MoE / 8-bit gate / 4-bit attn Qwen3.6 checkpoint..."
      "${MLX_VENV}/bin/python" "${SCRIPT_DIR}/scripts/build_qwen36_3bit_moe.py" \
        --src "$QWEN_SRC_DIR" \
        --dst "$QWEN_OUT_DIR"
    else
      hf_snapshot_download "$HF_QWEN_3BIT" "${MODELS_DIR}/Qwen3.6-35B-A3B-UD-MLX-3bit"
    fi
  fi
else
  log_info "Skipping Hugging Face model downloads (--skip-models)."
fi

# Step 6: Write active_profile.env and link m5-mlx-ctl
DEFAULT_QWEN_DIR="${MODELS_DIR}/Qwen3.6-35B-A3B-UD-MLX-3bit"
if [[ "$BUILD_QWEN_NVFP4" -eq 1 ]]; then
  DEFAULT_QWEN_DIR="${MODELS_DIR}/Qwen3.6-35B-A3B-NVFP4-MLX-3bit-MoE"
fi

if [[ ! -f "$ACTIVE_PROFILE" ]]; then
  cat > "$ACTIVE_PROFILE" <<EOF
# Active runtime profile for Apple M5 Air 24 GB MLX / oMLX server
M5_MLX_HOME="${M5_MLX_HOME}"
M5_MLX_REPO_DIR="${SCRIPT_DIR}"
M5_MLX_PORT="8080"
M5_MLX_UPSTREAM_PORT="8081"
M5_MLX_BACKEND="omlx"
M5_MLX_MODEL_ID="inkling-2-mini-oq3e-mtp"
M5_MLX_MODEL_DIR="${DEFAULT_QWEN_DIR}"
M5_MLX_CONTEXT_LENGTH="131072"
M5_MLX_KV_BITS="4"
M5_MLX_KV_GROUP_SIZE="64"
M5_MLX_PREFILL_STEP_SIZE="256"
M5_MLX_WIRED_LIMIT_MB="6144"
M5_MLX_KEEPWARM_INTERVAL_S="60"
M5_MLX_SSE_KEEPALIVE_S="5.0"
M5_MLX_OMLX_BIN="${OMLX_VENV}/bin/omlx"
M5_MLX_OMLX_MODEL_DIR="${OMLX_MODELS_DIR}"
M5_MLX_OMLX_DATA_DIR="${OMLX_DATA_DIR}"
M5_MLX_OMLX_SSD_CACHE_DIR="${OMLX_SSD_DIR}"
M5_MLX_OMLX_PORT="8081"
M5_MLX_OMLX_MEMORY_GUARD_GB="20.0"
M5_MLX_OVERFLOW_TOKENS="16500"
M5_MLX_RESTORE_IDLE_S="300"
M5_MLX_OVERFLOW_LLAMA_BIN="${M5_MLX_HOME}/bin/llama-server"
M5_MLX_OVERFLOW_MODEL_FILE="${MODELS_DIR}/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q3_K_XL.gguf"
M5_MLX_LLAMA_SERVER_BIN="${M5_MLX_HOME}/bin/llama-server"
M5_MLX_MODEL_FILE="${MODELS_DIR}/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q3_K_XL.gguf"
M5_MLX_DRAFT_MODEL_FILE="${MODELS_DIR}/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf"
M5_MLX_SPEC_DRAFT_MAX="0"
M5_MLX_LLAMA_PORT="8081"
M5_MLX_KV_CACHE_TYPE="q4_0"
M5_MLX_LLAMA_THREADS="6"
EOF
  log_info "Wrote default profile at ${ACTIVE_PROFILE}"
else
  log_info "Preserving existing profile at ${ACTIVE_PROFILE}"
fi

if [[ -f "${SCRIPT_DIR}/bin/m5-mlx-ctl" ]]; then
  chmod +x "${SCRIPT_DIR}/bin/m5-mlx-ctl"
  ln -sfn "${SCRIPT_DIR}/bin/m5-mlx-ctl" "${LOCAL_BIN_DIR}/m5-mlx-ctl"
  log_info "Linked CLI -> ${LOCAL_BIN_DIR}/m5-mlx-ctl"
else
  log_warn "${SCRIPT_DIR}/bin/m5-mlx-ctl not found; skipping CLI symlink."
fi

# Step 7: Optional LaunchAgent installation
if [[ "$ENABLE_LAUNCHAGENT" -eq 1 ]]; then
  if [[ -x "${SCRIPT_DIR}/bin/m5-mlx-ctl" ]]; then
    log_info "Installing LaunchAgent via m5-mlx-ctl..."
    M5_MLX_HOME="$M5_MLX_HOME" "${SCRIPT_DIR}/bin/m5-mlx-ctl" install-launchagent
  else
    PLIST_SRC="${SCRIPT_DIR}/launchd/com.community.m5-mlx-server.plist.template"
    PLIST_DST="$HOME/Library/LaunchAgents/com.community.m5-mlx-server.plist"
    mkdir -p "$HOME/Library/LaunchAgents"
    sed \
      -e "s|__M5_MLX_HOME__|${M5_MLX_HOME}|g" \
      -e "s|__M5_MLX_REPO_DIR__|${SCRIPT_DIR}|g" \
      -e "s|__M5_MLX_PORT__|8080|g" \
      -e "s|__M5_MLX_UPSTREAM_PORT__|8081|g" \
      "$PLIST_SRC" > "$PLIST_DST"
    launchctl bootout "gui/$(id -u)" "$PLIST_DST" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
    log_info "Installed and loaded LaunchAgent at ${PLIST_DST}"
  fi
fi

log_info "Installation complete! Ensure ${LOCAL_BIN_DIR} is in your PATH, then run:"
log_info "  m5-mlx-ctl doctor"
log_info "  m5-mlx-ctl start"
