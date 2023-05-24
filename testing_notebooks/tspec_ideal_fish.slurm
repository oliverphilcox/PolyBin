#!/bin/bash

#SBATCH --job-name=tl_fish
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=167:58:59
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p gen
#SBATCH --output=/mnt/home/ophilcox/out/polybin_tl_ideal_fish.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin

echo "Running option 1"
python -u testing_notebooks/tspec_ideal_fish.py 1

echo "Running option 2"
python -u testing_notebooks/tspec_ideal_fish.py 2
