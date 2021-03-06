#!/bin/bash

#SBATCH --job-name="dl3-drop"

#SBATCH --qos=training

#SBATCH --workdir=./

#SBATCH --output=./log_%j.out

#SBATCH --error=./log_%j.err

#SBATCH --ntasks=4

#SBATCH --gres gpu:1

#SBATCH --time=00:40:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python finetune.py VGG16_ImageNet
