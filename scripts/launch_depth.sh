#!/usr/bin/env bash
# Generate depth for a manifest on one GPU of a PBS job.
# usage: launch_depth.sh PBS_JOB_ID RUNTIME_ROOT MANIFEST IMAGES OUTPUT GPU_UUID
set -euo pipefail
job="$1"; runtime="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /opt/pbs/bin/pbs_attach -j "$job" /usr/bin/singularity exec --nv --cleanenv --bind /dev/shm \
  --env "CUDA_VISIBLE_DEVICES=$6" --env CUDA_DEVICE_ORDER=PCI_BUS_ID --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  /home/rintern01/containers/ubuntu22.sif "$runtime/venv-local/bin/python" -u "$root/scripts/generate_depth.py" \
  --manifest "$3" --images "$4" --output "$5" --cache "$runtime/source/caches/huggingface"
