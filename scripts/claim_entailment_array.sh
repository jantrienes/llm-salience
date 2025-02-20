#!/bin/bash

#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=8G

module purge
module load miniconda
source $CONDA_ROOT/bin/activate base
conda activate info-salience
export LC_ALL=C
export OMP_NUM_THREADS=1
export OUTLINES_CACHE_DIR=/tmp/outlines-$SLURM_JOB_ID

FACTS_PATH=$1
TASKS_PATH=$2

# Read the file path corresponding to this array job
task_file=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_PATH")

echo "Run claim entailment on summaries: $task_file"

python -m info_salience.claim_entailment \
    --facts_path "$FACTS_PATH" \
    --summaries_path "$task_file"
