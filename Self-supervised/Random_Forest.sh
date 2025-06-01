#!/bin/bash
#
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem=50GB
#SBATCH --partition=jobs-gpu
#SBATCH --account="account"
#SBATCH -o logs/slurm/jobs-gpu_%A_%a.log
#SBATCH -e logs/slurm/jobs-gpu_%A_%a.err
#SBATCH --array=1-3
#SBATCH -J "Name of the model"
#SBATCH --propagate=STACK

ulimit -s unlimited

echo "#########"
echo "Print the current environment (verbose)"
env

echo "#########"
echo "Show information on nvidia device(s)"
nvidia-smi


config_id=$((SLURM_ARRAY_TASK_ID - 1))

echo "Using config_id: $config_id"

singularity exec --nv --no-home --pwd $(pwd) -B "Define readable/writable paths" "path to container"/"name of container".sif python "path to python script"/Random_Forest.py --config_id $config_id
