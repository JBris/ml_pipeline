#!/bin/bash -e
##SBATCH --account=<id>
#SBATCH --job-name=distributed_ensemble_estimators  # job name (shows up in the queue)
#SBATCH --time=00:02:00      # Walltime (HH:MM:SS)
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=512M
#SBATCH --hint=multithread
##SBATCH --hint=nomultithread # disable hyperthreading that is activated by default
#SBATCH --tasks-per-node=1
#SBATCH --nodes=4
##SBATCH --reservation=test

let "worker_num=(${SLURM_NTASKS} - 1)"
total_cores=$((${worker_num} * ${SLURM_CPUS_PER_TASK}))
#total_cores=10
echo "Total cores: ${total_cores}"

suffix='6379'
ip_head=`hostname`:$suffix
export ip_head  

# Let's say the hostname=cluster-node-500 To view the dashboard on localhost:8265, set up an ssh-tunnel like this: (assuming the firewall allows it)
# $  ssh -N -f -L 8265:cluster-node-500:8265 user@big-cluster
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=`hostname` ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5

srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=`hostname` ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
sleep 5

python -u src/run.py ${total_cores} # Pass the total number of allocated CPUs
