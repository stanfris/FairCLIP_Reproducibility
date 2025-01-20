#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=CLIP_fairface
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32000M
#SBATCH --output=FairFace_CLIP_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Activate your environment
source activate fairclip

# Run your code
cd ../FairCLIP

DATASET_DIR=../data/fairface
RESULT_DIR=../results/fairface
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=20
SUMMARIZED_NOTE_FILE=fairface_label_train.csv
LR=0.001
BATCH_SIZE=64

PERF_FILE=${MODEL_ARCH}_FAIRFACE_CLIP.csv

python3 test.py