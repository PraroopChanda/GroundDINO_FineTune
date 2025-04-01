#!/bin/bash

# Set your MIG device here (replace with your real UUID from `nvidia-smi -L`)
export CUDA_VISIBLE_DEVICES=MIG-7f05759c-4e2b-5032-8031-32b9685a452d

# Activate conda env if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate groundingDINO1 
cd /scratch/user/praroop27/GroundingDINO

# (Optional) Confirm selected GPU
echo "Using device: $CUDA_VISIBLE_DEVICES"

# Run your Python script
python train_grounding_dino_kitti.py