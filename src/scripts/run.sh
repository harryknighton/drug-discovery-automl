#!/bin/bash
#SBATCH --nodes=1 --cpus-per-task 16
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH -J AutoML-Baseline
module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source ~/dev/torch_1_13_1/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m run_experiment --name=bla --dataset=AID1445 --epochs=100 --use-mf-pcba-splits