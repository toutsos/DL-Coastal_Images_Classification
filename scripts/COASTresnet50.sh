#!/bin/bash -l

#SBATCH --job-name=coast_resnet50
#SBATCH --output=./out/jobs/COASTresnet50_%j.out
#SBATCH --time=2:00:00
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=beards

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
--config-name COASTresnet50.yaml \
'hydra.run.dir=${paths.output_dir}' \
'datamodule.batch_size=64' \
'datamodule.num_workers=16' \
'module.lr=1e-4' \
'module.momentum=0.0' \
'module.nesterov=false' \
'module.weight_decay=0.001' \
'module.use_pretrained=true' \
'module.fine_tune=true' \
'module.patience=5' \
'module.use_saved_model=false' \
'trainer.num_nodes=1' \
'trainer.precision=32-true' \
'trainer.max_epochs=10' \
'trainer.accelerator=auto' \
'trainer.strategy=auto' \
'trainer.devices=auto' \
'trainer.sync_batchnorm=false' \
'trainer.gradient_clip_val=null' \
'trainer.gradient_clip_algorithm=norm' \
'trainer.profiler=simple' \
'callback_early_stopping.patience=20' \
'paths.output_dir='${OUTPUT_DIR} \
'paths.data_dir=/home/angelos.toutsios.gr/data/CS4321/HW1/teamsmt/data' \

