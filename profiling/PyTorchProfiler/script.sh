#!/bin/bash -l
#SBATCH --job-name=PyTprofilergpus
#SBATCH --account=<project_number>
#SBATCH --time=00:10:00     #wall-time 
#SBATCH --partition=accel   #partition 
#SBATCH --nodes=1           #nbr of nodes
#SBATCH --ntasks=1          #nbr of tasks
#SBATCH --ntasks-per-node=1 #nbr of tasks per nodes (nbr of cpu-cores, MPI-processes)
#SBATCH --cpus-per-task=1   #nbr of threads
#SBATCH --gpus=2            #total nbr of gpus
#SBATCH --mem=4G            #main memory
#SBATCH -o PyTprofilergpus.out  #slurm output 

# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

#define paths
Mydir=/cluster/work/users/<user_name>
MyContainer=${Mydir}/Container/pytorch_22.12-py3.sif
MyExp=${Mydir}/MyEx

#specify bind paths by setting the environment variable
#export SINGULARITY_BIND="${MyExp},$PWD"

#TF32 is enabled by default in the NVIDIA NGC TensorFlow and PyTorch containers 
#To disable TF32 set the environment variable to 0
#export NVIDIA_TF32_OVERRIDE=0

#to run singularity container 
srun singularity exec --nv -B ${MyExp},$PWD ${MyContainer} torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_PER_NODE:-2} ${MyExp}/resnet18_api_ddp.py

echo 
echo "--Job ID:" $SLURM_JOB_ID
echo "--total nbr of gpus" $SLURM_GPUS
echo "--nbr of gpus_per_node" $SLURM_GPUS_PER_NODE