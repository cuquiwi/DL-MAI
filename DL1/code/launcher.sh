#!/bin/bash

#SBATCH --job-name="DL1_SENNE"

#SBATCH --qos=training/debug

#SBATCH --workdir=.

#SBATCH --output=DL1_%j.out

#SBATCH --error=DL1_%j.err

#SBATCH --ntasks=4

#SBATCH --gres gpu:1

#SBATCH --time=00:02:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python CNN_Dropout_02.py
python CNN_Dropout_005.py
python CNN_Dropout_008.py
python CNN_Hidden512_add.py
python CNN_Hidden512_remove.py
