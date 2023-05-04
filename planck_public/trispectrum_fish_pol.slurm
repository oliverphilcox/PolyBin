#!/bin/bash

#SBATCH --job-name=trispectrum_fish_pol
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=95:58:59
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -p genx
#SBATCH --array=0-9
#SBATCH --output=/mnt/home/ophilcox/out/trispectrum_fish_pol%a.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin
index=$SLURM_ARRAY_TASK_ID

echo Running simulation $index
python -u planck_public/trispectrum_fish_pol.py $index
