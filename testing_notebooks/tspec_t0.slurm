#!/bin/bash

#SBATCH --job-name=tspec_t0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=95:58:59
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p genx
#SBATCH --output=/mnt/home/ophilcox/out/polybin_t0.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin

python -u testing_notebooks/tspec_t0.py