#!/bin/bash
#SBATCH --job-name=lightgcn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%j.out
# Usage (from BookRecommendation/):  sbatch slurm/train_gpu.sbatch [extra train.py args]
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
.venv-gpu/bin/python -m lightgcn.train --device cuda "$@"
