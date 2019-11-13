#!/bin/bash

#SBATCH --job-name="DL2_SENNE"

#SBATCH -D .

#SBATCH --workdir=.

#SBATCH --output=DL1_%j.out

#SBATCH --error=DL1_%j.err

#SBATCH --ntasks=40

#SBATCH --gres gpu:1

#SBATCH --time=20:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python LSTM.py
