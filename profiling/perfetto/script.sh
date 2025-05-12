#!/bin/bash
#SBATCH --account=your_project_number
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00


module purge
module use /appl/local/training/modules/AI-20241126/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

srun singularity exec $CONTAINER torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_PER_NODE:-4} ./ddp_train.py --num_workers=4
