#!/bin/bash

#SBATCH --job-name=trispectrum_num
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --time=47:58:59
#SBATCH --mem=500gb
#SBATCH --cpus-per-task=40
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --array=0-0
#SBATCH --output=/mnt/home/ophilcox/out/trispectrum_num_pol%a.log

# Run 1 job per CPU
N_JOB=10                # create as many jobs as tasks

# Load modules
module load modules/2.1 openmpi python-mpi intel-mkl cfitsio
source ~/mpivenv2/bin/activate

cd /mnt/home/ophilcox/PolyBin

for((i=0;i<$N_JOB;i++))
do
  (( task_id = SLURM_ARRAY_TASK_ID * N_JOB + i ))

  echo "Analyzing simulation $task_id"

  python -u planck_public/trispectrum_num_pol.py $task_id &

done

#Wait for all
wait

echo
echo "All done!"
