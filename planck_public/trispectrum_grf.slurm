#!/bin/bash

#SBATCH --job-name=trispectrum_grf
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=23:58:59
#SBATCH --mem=300gb
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --array=100-999
#SBATCH --output=/home/ophilcox/out/trispectrum_grf%a.log

# Load modules
module load anaconda3/2021.11
source activate sbi

cd /home/ophilcox/cmb-4pt
index=$SLURM_ARRAY_TASK_ID

echo Running simulation $index
python -u planck/trispectrum_grf.py $index
