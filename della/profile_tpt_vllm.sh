#!/bin/bash
#SBATCH --job-name=profile_tpt_vllm_72b              # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="h100"
#SBATCH --time=0:30:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
##SBATCH --partition=pli
##SBATCH --account=specreason
#SBATCH --partition=pli-lc
#SBATCH --account=ravi-group


# CLUSTER="ravi"
CLUSTER="della"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "ravi" ]; then
    DATA_DIR="/home/ruipan/data2"
elif [ "$CLUSTER" = "della" ]; then
    DATA_DIR="/scratch/gpfs/RAVIAN/rp2773/data"
    export HF_HOME="/scratch/gpfs/RAVIAN/rp2773/hf_cache"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    source /scratch/gpfs/RAVIAN/rp2773/miniconda3/etc/profile.d/conda.sh
    nvidia-smi
else
    echo "Error: CLUSTER must be either 'ravi' or 'della'"
    exit 1
fi
conda activate vllm_dllm

OUTPUT_DIR="${DATA_DIR}/diffspec"

# TARGET_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
# TARGET_MODEL="Qwen/Qwen2.5-32B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-72B-Instruct"
TP_SIZE=2


# Function to check if a server is ready
wait_for_server() {
    local port=$1
    while true; do
        # Try to connect to the server
        curl -s http://localhost:$port/get_model_info > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done
}

# launch 32b model and 1.5b model one by one
vllm serve "$TARGET_MODEL" --dtype auto -tp "$TP_SIZE" --max_model_len 2000 --gpu-memory-utilization 0.95 --enable-prefix-caching --port 30000 --enforce-eager &
VLLM_BASE_PID=$!
wait_for_server 30000
nvidia-smi




echo "Profiling TPT for vLLM on model ${TARGET_MODEL} ..."
python ../profiling/profile_tpt_vllm.py > "${OUTPUT_DIR}/logs/profile_tpt_h100_${TARGET_MODEL//\//_}.log"

