#!/bin/bash

#SBATCH --job-name=trispectrum_fish
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=47:58:59
#SBATCH --mem=200gb
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --array=0-99
#SBATCH --output=/home/ophilcox/out/trispectrum_fish%a.log

# Load modules
module load anaconda3/2021.11
source activate sbi

cd /home/ophilcox/cmb-4pt
index=$SLURM_ARRAY_TASK_ID

echo Running simulation $index
python -u planck_public/trispectrum_fish.py $index
