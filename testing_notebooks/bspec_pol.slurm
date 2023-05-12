#!/bin/bash

#SBATCH --job-name=bl_test
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=95:58:59
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --array=0-100
#SBATCH --output=/mnt/home/ophilcox/out/polybin_bl%a.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin

echo "Running option 1"
python -u testing_notebooks/bspec_pol.py 1 $SLURM_ARRAY_TASK_ID

echo "Running option 2"
python -u testing_notebooks/bspec_pol.py 2 $SLURM_ARRAY_TASK_ID

echo "Running option 3"
python -u testing_notebooks/bspec_pol.py 3 $SLURM_ARRAY_TASK_ID
