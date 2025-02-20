#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --partition=owner_fb12

module purge
module load miniconda
source $CONDA_ROOT/bin/activate base
conda activate info-salience
export LC_ALL=C
export OMP_NUM_THREADS=1
export OUTLINES_CACHE_DIR=/tmp/outlines-$(date +%s | sha256sum | base64 | head -c 10)

python -m info_salience.summarization_baselines "$@"
