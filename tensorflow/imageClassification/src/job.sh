#!/bin/bash
#SBATCH --account=project_number
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

module purge
module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems singularity-CPEbits
# Point to scratch
HOST_SCRATCH="/scratch/project_number/binod/tensorflow"
MODEL_SAVE_DIR="$HOST_SCRATCH/saved_models"
LOG_DIR="$HOST_SCRATCH/logs"

# Create directories (if missing)
mkdir -p $MODEL_SAVE_DIR $LOG_DIR
CONTAINER=/appl/local/containers/sif-images/lumi-tensorflow-rocm-6.2.0-python-3.10-tensorflow-2.16.1-horovod-0.28.1.sif
srun singularity exec --bind $HOST_SCRATCH:/mnt $CONTAINER bash -c "source /opt/miniconda3/bin/activate tensorflow &&  python train.py \
  --model_save_path /mnt/saved_models/cifar10_cnn.h5 \
  --log_dir /mnt/logs "

# Note: though we have defined the software needed for us in the env.yml, we are using the pre-built container
# for this particular example.Also on this particular instance, we had to source this specific path(/opt/miniconda3/bin/) 
# where the tensorflow was installed