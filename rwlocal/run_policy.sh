#!/usr/bin/env bash
set -euo pipefail

# Same input style as RealWorldInference/run_policy.sh.
# Implementation style follows PhySeek/README.md:
#   physeek run <module>:<PolicyClass> --id <policy_id> --sep <wss-url>

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage:
  ./rwlocal/run_policy.sh <gpuid> <sep_env> <robot_type> <policy_id> <checkpoint>

Example:
  ./rwlocal/run_policy.sh 0 r6000 xingchen runqing /home/ubuntu/newproject/checkpoints/xingchen_shujia/ckpts/step-20000-ema.safetensors
EOF
}

fail() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 5 ]]; then
  fail "expected 5 arguments."
fi

GPU_ID="$1"
SEP_ENV="$2"
ROBOT_TYPE="$3"
POLICY_ID="$4"
CHECKPOINT="$5"

case "${ROBOT_TYPE}" in
  xingchen|moz) ;;
  *) fail "invalid robot_type '${ROBOT_TYPE}'. Expected: xingchen or moz." ;;
esac

case "${SEP_ENV}" in
  r6000)
    SEP_URL="wss://172.25.1.62:9668"
    TLS_ARGS=(--insecure)
    ;;
  *)
    fail "invalid sep_env '${SEP_ENV}'. Expected: r6000."
    ;;
esac

echo "Starting RWI Falcon policy via PhySeek SDK:"
echo "  gpuid:       ${GPU_ID}"
echo "  sep_env:     ${SEP_ENV}"
echo "  robot_type:  ${ROBOT_TYPE}"
echo "  policy_id:   ${POLICY_ID}"
echo "  checkpoint:  ${CHECKPOINT}"
echo "  sep_url:     ${SEP_URL}"
echo

CUDA_VISIBLE_DEVICES="${GPU_ID}" physeek run rwlocal.rwi_falcon_policy:RwiFalconPolicy \
  --id "${POLICY_ID}" \
  --sep "${SEP_URL}" \
  --policy-type falcon \
  --policy-arg robot_type="${ROBOT_TYPE}" \
  --policy-arg checkpoint_path="${CHECKPOINT}" \
  --policy-arg return_video=true \
  "${TLS_ARGS[@]}"
