#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 16
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J AutoML-Baseline
#SBATCH --account=su114-gpu
#SBATCH --output "slurm-%x-%j.out"

module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source /home/h/hjk51/dev/torch_1_13_1/bin/activate

date;

CUDA_VISIBLE_DEVICES=0 python -m src.scripts.run_experiment \
  optimise \
  --name baseline \
  --dataset "$1" \
  --search-space simple \
  --max-evaluations 100 \
;

date;