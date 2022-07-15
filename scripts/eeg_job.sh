#! /bin/bash
#SBATCH --account=def-xxxx
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=2,1,0
#SBATCH --time=0-5:00:00
#SBATCH --job-name=computing
#SBATCH --output=%j-computing.out
#SBATCH --error=%j-computing.err
#SBATCH --mail-user=xxxxx
#SBATCH --mail-type=ALL

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
cd /home/xxxx/scratch/data-imbalance
pip install .
python -u scripts/eeg_imbalance_analysis.py $SLURM_ARRAY_TASK_ID