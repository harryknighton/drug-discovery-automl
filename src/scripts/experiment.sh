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

CUDA_VISIBLE_DEVICES=0 python -m src.scripts.run_experiment \
  experiment \
  --name baseline \
  --dataset "$1" \
  --dataset-usage "$2" \
  --epochs 200 \
  --use-mf-pcba-splits \
  --precision=medium \
  --seeds 4281 7945 7026 \
  --num-layers 2 \
  --layer-types GCN GIN GAT GATv2 \
  --features 128 \
  --pooling-functions ADD MEAN MAX \
  --num-regression-layers 2 \
  --regression-features 128 \
;

date;