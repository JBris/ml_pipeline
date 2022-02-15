#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --account=ptec03219
#SBATCH --job-name=interpret_ml
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --time=23:59:59
#SBATCH --hint=multithread

set -x

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Module load
module load  Python/3.9.9-gimkl-2020a

port=6379
RAY_IP_HEAD=${head_node_ip}:${port}
export RAY_IP_HEAD
echo "IP Head: $RAY_IP_HEAD"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block & # --num-gpus "${SLURM_GPUS_PER_TASK}"

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
   node_i=${nodes_array[$i]}
   echo "Starting WORKER $i at $node_i"
   srun --nodes=1 --ntasks=1 -w "$node_i" \
       ray start --address "$RAY_IP_HEAD" \
       --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
   sleep 5
done

. ../../.env
. ../../env_hook.sh

export RAY_ADDRESS=$RAY_IP_HEAD
export DISABLE_PLOTLY=1

# Vars
# Override using sbatch --export=base_dir=.,scenario=.,...,from_params=. tpot_automl.sl
base_dir="${SIZING_DIR:-.}"
scenario="${ML_PIPELINE_SCENARIO:-.}"
from_params="${CONFIG_FILE:-.}"

python -u src/run.py --base_dir "${base_dir}" --scenario "${scenario}" --from_params "${from_params}"
