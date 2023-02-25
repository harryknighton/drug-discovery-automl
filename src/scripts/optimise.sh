#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 32
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH -J automl-baseline
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
  --dataset-usage "$2" \
  --search-space simple \
  --max-evaluations 300 \
  --seed 2409 \
  --num-workers 32 \
  --precision medium \
;

date;