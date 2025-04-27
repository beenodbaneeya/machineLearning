#!/bin/bash
#SBATCH --job-name=tf_benchmark
#SBATCH --account=project_number
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:45:00

module purge
module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems singularity-CPEbits

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=ERROR                        

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500


# Point to scratch
HOST_SCRATCH="/scratch/project_number/binod/tf-mirror"
MODEL_SAVE_DIR="$HOST_SCRATCH/saved_models"
LOG_DIR="$HOST_SCRATCH/logs"




# Create directories (if missing)
mkdir -p $LOG_DIR/{cpu,gpu} $MODEL_SAVE_DIR

CONTAINER=/appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif



### MAIN BENCHMARK RUN ###
echo "=== Proceeding with Main Benchmark ==="

srun --export=ALL singularity exec --bind $HOST_SCRATCH:/mnt $CONTAINER \
bash -c "
  export RANK=\$SLURM_PROCID; \
  export LOCAL_RANK=\$SLURM_LOCALID; \
  source /opt/miniconda3/bin/activate tensorflow; \
  python benchmark.py \
    --model_save_dir /mnt/saved_models \
    --log_dir /mnt/logs
"