#!/bin/bash -l

#SBATCH --job-name=coast_resnet50
#SBATCH --output=./out/jobs/COASTvgg16_%j.out
#SBATCH --time=2:00:00
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=beards,genai

# NOTES:
# SBATCH --ntasks-per-node and --gres, training.devices need to be the same
# SBATCH --cpus-per-task and datamodule.num_workers should be the same
# It appears that specifying 'srun' before 'python' is necessary
# You need to re-specify --time to srun, or else your job will be killed after a short amount of time
# If you want to run in debug mode, run single GPU

OUTPUT_DIR=out/$(date +%Y-%m-%d_%H-%M-%S)

mkdir -p ${OUTPUT_DIR}

## Pixi ##
eval "$(pixi shell-hook -s bash)"

## Debugging ##
export HYDRA_FULL_ERROR=1

## Multi-node training ##

# InfiniBand
# export NCCL_IB_HCA=ib0

# If InfiniBand is having issues
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

#srun --time=2:00:00 python -m debugpy --wait-for-client --listen 0.0.0.0:54327 src/train.py \
srun --time=2:00:00 python src/train.py \
--config-dir configs \
--config-name COASTvgg.yaml \
'hydra.run.dir=${paths.output_dir}' \
'datamodule.batch_size=64' \
'datamodule.num_workers=3' \
'datamodule.random_rotate=10' \
'module.lr=0.001' \
'module.momentum=0.0' \
'module.nesterov=false' \
'module.weight_decay=0.0001' \
'module.patience=10' \
'module.frozen_layers=[0,2,5,7,10,12]' \
'module.use_pretrained=true' \
'module.fine_tune=true' \
'module.load_ckpt=true' \
'module.ckpt_path=/home/jarrad.singley/data/deeplearning/DL-Coastal_Images_Classification/out/2025-02-04_09-36-29_ff_best/lightning_logs/version_0/checkpoints/last.ckpt' \
'module.dropout=0.5' \
'trainer.num_nodes=1' \
'trainer.precision=32-true' \
'trainer.max_epochs=30' \
'trainer.accelerator=auto' \
'trainer.strategy=auto' \
'trainer.devices=auto' \
'trainer.sync_batchnorm=false' \
'trainer.gradient_clip_val=null' \
'trainer.gradient_clip_algorithm=norm' \
'trainer.profiler=simple' \
'paths.output_dir='${OUTPUT_DIR} \
'paths.data_dir=/home/jarrad.singley/data/deeplearning/DL-Coastal_Images_Classification/data' \
