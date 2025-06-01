#!/bin/bash
#
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=70GB
#SBATCH --partition=jobs-gpu
#SBATCH --account="account"
#SBATCH -o logs/slurm/jobs-gpu_%A_%a.log
#SBATCH -e logs/slurm/jobs-gpu_%A_%a.err
#SBATCH --array=1-"Number of configs"%1
#SBATCH -J "Name of the model"
#SBATCH --propagate=STACK

ulimit -s unlimited

echo "#########"
echo "Print the current environment (verbose)"
env

echo "#########"
echo "Show information on nvidia device(s)"
nvidia-smi

file=$(ls "Path to configs"/Configs/"Name of the model"/ | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)
echo "parsing config: "$file

for i in {0..2}; do
  echo "Iteration $i"
  singularity exec --nv --no-home --pwd $(pwd) -B "Define readable/writable paths" "path to container"/"name of container".sif python "path to python script"/"name of python script".py -c "Path to configs"/Configs/"Name of the model"/$file --config_id $i
done