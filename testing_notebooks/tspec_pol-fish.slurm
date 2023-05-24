#!/bin/bash

#SBATCH --job-name=tl_test_fish
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=95:58:59
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --mem 700gb
#SBATCH -p cca
#SBATCH --array=0-9
#SBATCH --output=/mnt/home/ophilcox/out/polybin_tl_fish%a.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
module add disBatch
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin

echo "Running option 1"
python -u testing_notebooks/tspec_pol_fish.py 1 $SLURM_ARRAY_TASK_ID

echo "Running option 2"
python -u testing_notebooks/tspec_pol_fish.py 2 $SLURM_ARRAY_TASK_ID