#!/bin/bash -l

#SBATCH --job-name=coast_vit_finetune
#SBATCH --output=./out/jobs/coast_vit_finetune-%j.out
#SBATCH --time=10:00:00
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

## Pixi ##
eval "$(pixi shell-hook -s bash)"

## Debugging ##
export HYDRA_FULL_ERROR=1

# Set fixed parameters
datamodule_batch_size=32
datamodule_rotation=20
module_lr=0.001
module_momentum=0.7
module_weight_decay=0.0001
module_dropout=0.6
module_n_layer1=512
module_n_layer2=128
module_batch_norm=true
module_unfreeze_depth=6
CHECKPOINT_DIR=/home/roberto.milanese/DeepLearning/DL-Coastal_Images_Classification_2/out/bests/vit_b_frzn_2025-zzz_best_submitting/lightning_logs/version_0/checkpoints/last.ckpt


# Create output directory
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR=out/bests/v_bit_ft_$TIMESTAMP
mkdir -p ${OUTPUT_DIR}

# Run the training script
srun --time=2:00:00 python src/train.py \
    --config-dir configs \
    --config-name COAST_vit_b_16_unfrozen.yaml \
    "hydra.run.dir=${OUTPUT_DIR}" \
    "datamodule.batch_size=${datamodule_batch_size}" \
    "datamodule.rotation=${datamodule_rotation}" \
    "module.lr=${module_lr}" \
    "module.momentum=${module_momentum}" \
    "module.weight_decay=${module_weight_decay}" \
    "module.n_layer1=${module_n_layer1}" \
    "module.n_layer2=${module_n_layer2}" \
    "module.dropout=${module_dropout}" \
    "module.use_pretrained=false" \
    "module.batch_norm=${module_batch_norm}" \
    "module.unfreeze_depth=${module_unfreeze_depth}" \
    "module.fine_tune=true" \
    "trainer.max_epochs=27" \
    "module.checkpoint=${CHECKPOINT_DIR}" \
    "paths.output_dir=${OUTPUT_DIR}" \
    "paths.data_dir=/home/roberto.milanese/HW1/"
