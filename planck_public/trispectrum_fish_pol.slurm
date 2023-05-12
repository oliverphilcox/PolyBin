#!/bin/bash

#SBATCH --job-name=trispectrum_fish_pol
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=95:58:59
#SBATCH --mem=3000gb
#SBATCH -N 1
#SBATCH --cpus-per-task=90
#SBATCH -p mem
#SBATCH --output=/mnt/home/ophilcox/out/trispectrum_fish_pol%a.log

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin
index=$SLURM_ARRAY_TASK_ID

for((i=1;i<=10;i++))
do
  echo "Analyzing simulation $task_id"

  python -u planck_public/trispectrum_fish_pol.py $i

done