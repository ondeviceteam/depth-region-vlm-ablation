#!/usr/bin/env bash
# Run the latency protocol on ONE GPU of a PBS job, one process at a time (no other work on that GPU).
# usage: launch_latency.sh PBS_JOB_ID RUNTIME_ROOT DATA_ROOT EXPERIMENT OUTPUT GPU_INDEX
set -euo pipefail
job="${1:?PBS job ID}"; runtime="${2:?runtime root}"; data="${3:?data root}"; experiment="${4:?experiment}"; output="${5:?output dir}"; gpu="${6:?GPU index in the job}"
if [[ "${EVAL_PBS_ATTACHED:-0}" != 1 ]]; then
  exec /opt/pbs/bin/pbs_attach -j "$job" /usr/bin/env EVAL_PBS_ATTACHED=1 /bin/bash "$0" "$@"
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$root/../ToSA"
git -C "$upstream" diff --exit-code HEAD -- llava ToSA
git -C "$upstream" diff 30767ed473dc919137cbeb9d4baa6a2f2e3c44f1 HEAD -- llava ToSA \
  | cmp - "$root/experiments/$experiment/upstream_compatibility.patch"
source "/var/spool/pbs/aux/$job.env"
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$output"
run() {
  /usr/bin/singularity exec --nv --cleanenv --bind /dev/shm \
    --env "CUDA_VISIBLE_DEVICES=${gpus[$gpu]}" --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONHASHSEED=42 \
    --env "HF_HOME=$runtime/tosa-official/huggingface" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env "PBS_JOBID=$job" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env TOKENIZERS_PARALLELISM=false \
    /home/rintern01/containers/ubuntu22.sif "$@"
}
run "$runtime/venv-local/bin/python" -u "$root/scripts/run_depth_latency.py" --experiment "$experiment" --data "$data" \
  --cache "$runtime/source/caches/huggingface" --output "$output" > "$output/depth.log" 2>&1
for model in 0.5B 7B; do
  for variant in 2D 3D; do
    run "$runtime/tosa-official/venv/bin/python" -u "$root/scripts/run_latency.py" --experiment "$experiment" --model "$model" \
      --variant "$variant" --data "$data" --upstream "$upstream" --source-repo "$runtime/source-v5.1" \
      --runtime "$runtime/tosa-official" --output "$output" > "$output/${model}_${variant}.log" 2>&1
  done
done
echo LATENCY_DONE
