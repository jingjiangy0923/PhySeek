#!/usr/bin/env bash
# 同步 PhySeek 到 r6000；只拷贝文件，不启动服务。
# 用法: bash deploy.sh r6000

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_target() {
  case "$1" in
    r6000) SSH_HOST=r6000 DIR=/home/ubuntu/newproject/ss-embodied/PhySeek ;;
    *) return 1 ;;
  esac
}

usage() {
  cat <<'EOF'
用法: bash deploy.sh <target>

可选 target:
  r6000

环境变量覆盖（可选）:
  REMOTE_SSH_HOST
  REMOTE_DIR
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="$1"
if ! get_target "${TARGET}"; then
  echo "未知 target: ${TARGET}" >&2
  echo >&2
  usage >&2
  exit 2
fi

REMOTE_SSH_HOST="${REMOTE_SSH_HOST:-${SSH_HOST}}"
REMOTE_DIR="${REMOTE_DIR:-${DIR}}"

echo "==> 清理本地 Python 缓存 ..."
python - "${ROOT}" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
removed = 0
for path in root.rglob("*"):
    if path.is_dir() and (path.name in names or path.name.endswith(".egg-info")):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
print(f"    已删除 {removed} 个缓存/构建目录" if removed else "    无缓存/构建目录")
PY

RSYNC_LOG="$(mktemp)"
SSH_CTL="${TMPDIR:-/tmp}/.physeek-deploy-cm-$$"
SSH_OPTS="-o ConnectTimeout=10 -o ControlMaster=auto -o ControlPath=${SSH_CTL} -o ControlPersist=30"

cleanup() {
  ssh -o ControlPath="${SSH_CTL}" -O exit "${REMOTE_SSH_HOST}" 2>/dev/null || true
  rm -f "${RSYNC_LOG}"
}
trap cleanup EXIT

printf -v REMOTE_DIR_Q "%q" "${REMOTE_DIR}"

echo "==> 同步 PhySeek [${TARGET}] → ${REMOTE_SSH_HOST}:${REMOTE_DIR}"
echo "==> 创建远端目录 ..."
if ! ssh ${SSH_OPTS} "${REMOTE_SSH_HOST}" "mkdir -p -- ${REMOTE_DIR_Q}"; then
  echo "ERROR: 创建远端目录失败" >&2
  exit 1
fi

echo "==> 开始 rsync ..."
if rsync -av --delete --stats \
  -e "ssh ${SSH_OPTS}" \
  --exclude='.git'            \
  --exclude='.DS_Store'       \
  --exclude='__pycache__'     \
  --exclude='*.py[cod]'       \
  --exclude='.pytest_cache'   \
  --exclude='.mypy_cache'     \
  --exclude='.ruff_cache'     \
  --exclude='*.egg-info'      \
  --exclude='.venv'           \
  --exclude='venv'            \
  --exclude='logs'            \
  --exclude='outputs'         \
  --exclude='runs'            \
  --exclude='wandb'           \
  --exclude='checkpoints'     \
  --exclude='*.ckpt'          \
  --exclude='*.pt'            \
  --exclude='*.pth'           \
  --exclude='*.safetensors'   \
  --exclude='*.mp4'           \
  --exclude='*.avi'           \
  --exclude='*.mov'           \
  "${ROOT}/" "${REMOTE_SSH_HOST}:${REMOTE_DIR}/" >"${RSYNC_LOG}" 2>&1; then
  :
else
  rsync_status=$?
  python - "${RSYNC_LOG}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
print("\n".join(path.read_text(errors="replace").splitlines()[:200]))
PY
  echo "ERROR: 文件同步失败，rsync exit_code=${rsync_status}" >&2
  exit "${rsync_status}"
fi

transferred_files="$(python - "${RSYNC_LOG}" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors="replace")
for line in text.splitlines():
    if "Number of regular files transferred" in line or "Number of files transferred" in line:
        value = line.split(":", 1)[-1]
        value = re.sub(r"[,\s]", "", value)
        print(value or "0")
        break
else:
    print("0")
PY
)"

echo ""
echo "==> 同步完成 [${TARGET}]，成功上传文件数：${transferred_files}（仅拷贝文件，未启动服务）"
