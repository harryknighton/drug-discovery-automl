#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 16
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -J AutoML-Baseline
#SBATCH --account=su114-gpu

module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source /home/h/hjk51/dev/torch_1_13_1/bin/activate

date;

CUDA_VISIBLE_DEVICES=0 python -m src.scripts.run_experiment -N baseline -D AID1445 -e 200 --use-mf-pcba-splits --precision=medium -s 7339 2263 7272 -n 3 -l GCN -f 64 128 256 512 -p ADD;

date;