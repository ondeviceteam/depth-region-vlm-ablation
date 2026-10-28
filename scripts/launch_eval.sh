#!/usr/bin/env bash
# Launch run_eval.py workers on the two GPUs of a PBS job.
# usage: launch_eval.sh PBS_JOB_ID RUNTIME_ROOT DATA_ROOT EXPERIMENT OUTPUT WORKER...
#   EXPERIMENT = directory name under experiments/ (e.g. 2026-10-28_0.5b_vqav2_gqa)
#   WORKER = dataset:variant:shard:num_shards:gpu_index   (e.g. gqa:2D:0:4:0)
set -euo pipefail
job="${1:?PBS job ID}"; runtime="${2:?runtime root}"; data="${3:?data root}"; experiment="${4:?experiment}"; output="${5:?output dir}"; shift 5
if [[ "${EVAL_PBS_ATTACHED:-0}" != 1 ]]; then
  exec /opt/pbs/bin/pbs_attach -j "$job" /usr/bin/env EVAL_PBS_ATTACHED=1 /bin/bash "$0" "$job" "$runtime" "$data" "$experiment" "$output" "$@"
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$root/../ToSA"
git -C "$upstream" diff --exit-code HEAD -- llava ToSA
git -C "$upstream" diff 30767ed473dc919137cbeb9d4baa6a2f2e3c44f1 HEAD -- llava ToSA \
  | cmp - "$root/experiments/$experiment/upstream_compatibility.patch"
source "/var/spool/pbs/aux/$job.env"
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$output"
pids=()
for worker in "$@"; do
  IFS=':' read -r dataset variant shard shards gpu <<< "$worker"
  case "$dataset" in
    vqav2_count) images="$data/coco/val2014"; depth="$data/coco/depth" ;;
    gqa) images="$data/gqa/images"; depth="$data/gqa/depth" ;;
  esac
  /usr/bin/singularity exec --nv --cleanenv --bind /dev/shm \
    --env "CUDA_VISIBLE_DEVICES=${gpus[$gpu]}" --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONHASHSEED=42 \
    --env "HF_HOME=$runtime/tosa-official/huggingface" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env "PBS_JOBID=$job" --env OMP_NUM_THREADS=3 --env MKL_NUM_THREADS=3 --env TOKENIZERS_PARALLELISM=false \
    /home/rintern01/containers/ubuntu22.sif "$runtime/tosa-official/venv/bin/python" -u "$root/scripts/run_eval.py" \
    --experiment "$experiment" --dataset "$dataset" --variant "$variant" --shard "$shard" --num-shards "$shards" \
    --images "$images" --depth "$depth" --upstream "$upstream" --source-repo "$runtime/source-v5.1" \
    --runtime "$runtime/tosa-official" --output "$output" \
    > "$output/${dataset}_${variant}_shard${shard}of${shards}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "WORKERS_FINISHED status=$status"
exit "$status"
