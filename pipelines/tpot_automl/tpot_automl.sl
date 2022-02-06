#!/bin/bash -e
#SBATCH --account=ptec03219
#SBATCH --job-name=tpot_automl   
#SBATCH --time=23:59:59
#SBATCH --hint=multithread
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512M
#SBATCH --hint=multithread

# Activate conda environment
module load Anaconda3/2019.03-gimkl-2018b
. $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /nesi/project/ptec03219/tpot_dask

. ../../.env

# Vars
# Override using sbatch --export=base_dir=.,scenario=.,...,from_params=. tpot_automl.sl
base_dir="${SIZING_DIR:-.}"
scenario="${PIPELINE_SCENARIO:-.}"
from_params="${CONFIG_FILE:-.}"

srun python src/run.py --base_dir "${base_dir}" --scenario "${scenario}" --from_params "${from_params}"
