#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=trienes@staff.uni-marburg.de
#SBATCH --mail-type=END,FAIL

module purge
module load miniconda
source $CONDA_ROOT/bin/activate base
conda activate info-salience
export LC_ALL=C
export OMP_NUM_THREADS=1
export OUTLINES_CACHE_DIR=/tmp/outlines-$SLURM_JOB_ID

python -m info_salience.qa "$@"
