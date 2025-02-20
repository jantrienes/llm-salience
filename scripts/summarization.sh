#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --partition=owner_fb12

module purge
module load miniconda
source $CONDA_ROOT/bin/activate base
conda activate info-salience
export LC_ALL=C
export OMP_NUM_THREADS=1
export OUTLINES_CACHE_DIR=/tmp/outlines-$(date +%s | sha256sum | base64 | head -c 10)

# Set this for tensor_parallel_size >= 2, See: https://github.com/vllm-project/vllm/issues/6152
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Calculate temperature in a slurm array job
    temperature=$(echo "(1/($SLURM_ARRAY_TASK_COUNT-1)) * $SLURM_ARRAY_TASK_ID" | bc -l)
    python -m info_salience.summarization --temperature $temperature "$@"
else
    python -m info_salience.summarization "$@"
fi
