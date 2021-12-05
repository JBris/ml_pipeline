#!/bin/bash -e
##SBATCH --account=<ID>
#SBATCH --job-name=dask_test  # job name (shows up in the queue)
#SBATCH --time=06:00:00      # Walltime (HH:MM:SS)
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512M
#SBATCH --hint=multithread
##SBATCH --hint=nomultithread # disable hyperthreading that is activated by default

#module purge
module load Anaconda3/2019.03-gimkl-2018b
. $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /nesi/project/<ID>/daskenv

srun python dask_example.py
