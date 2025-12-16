#!/bin/bash
#SBATCH --job-name=chess-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

module load cuda || true
source ~/venv/bin/activate

export OMP_NUM_THREADS=32

python train.py
