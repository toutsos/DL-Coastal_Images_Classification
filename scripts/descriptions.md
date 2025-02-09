# Project Script Descriptions

## 1. `best_mnv2_frzn_rerun.sh`
This script trains a MobileNetV2 model with a frozen feature extractor, replicating the best performing model using the same hyperparameters. Its output directory is out/bests/mnv2_frzn_$TIMESTAMP and data dir is /home/roberto.milanese/HW1/ which will need to be changed to your own data directory.

## 2. `best_mnv2_ft.sh`
This script fine-tunes a MobileNetV2 model, which was previously trained in a frozen state. It loads the last checkpoint from a previous frozen training session and unfreezes 12 layers for further tuning. It uses the same hyperparameters as the best run indicated in the submission write up. The data dir is /home/roberto.milanese/HW1/ which will need to be changed to your own data directory, and checkpoint path is /home/roberto.milanese/DeepLearning/DL-Coastal_Images_Classification_2/out/bests/mnv2_frzn_2025-02-06_21-08-11/lightning_logs/version_0/checkpoints/last.ckpt, which will also need to be changed to your own checkpoint path.

## 3. `best_vit_b_frz_rerun.sh`
This script trains a Vision Transformer (ViT-B/16) with a frozen feature extractor. The model uses pretrained weights and does not allow fine-tuning. It pulls data from the `/home/roberto.milanese/HW1/` directory and outputs to out/bests/vit_b_frzn_$TIMESTAMP. The data dir will need to be changed to your own data directory.

## 4. `best_v_bit_ft_rerun.sh`
This script fine-tunes a checkpoint from a previous Vision Transformer (ViT-B/16) training session. It unfreezes the last 6 layers of the model and uses the same hyperparameters as the best run indicated in the submission write up. The data dir is /home/roberto.milanese/HW1/ which will need to be changed to your own data directory, and checkpoint path is CHECKPOINT_DIR=/home/roberto.milanese/DeepLearning/DL-Coastal_Images_Classification_2/out/bests/vit_b_frzn_2025-zzz_best_submitting/lightning_logs/version_0/checkpoints/last.ckpt, which will also need to be changed to your own checkpoint path.

## 5. `COASTvgg16.sh`
This script fine-tunes a VGG-16 model for coastal image classification. It loads a checkpoint from a previous run and unfreezes multiple layers (`[0,2,5,7,10,12]`). It loads the checkpoint from 'module.ckpt_path=/home/jarrad.singley/data/deeplearning/DL-Coastal_Images_Classification/out/2025-02-04_09-36-29_ff_best/lightning_logs/version_0/checkpoints/last.ckpt' and loads data from 'paths.data_dir=/home/jarrad.singley/data/deeplearning/DL-Coastal_Images_Classification/data', outputting to out/$(date +%Y-%m-%d_%H-%M-%S). The data dir and checkpoint path will need to be changed to your own data directory and checkpoint path.

##6. `COASTresnet50.sh`
This script trains a Resnet50 model for coastal image classification. You can change most of the hyperparameters through this script, and if a hyperparam doesn't exist right now, it can be explicity added.
- For **Fine-tune** a model use the flag: `module.fine_tune=true` else it will trained with **Fixed-features**.
- For loading an already existing model and train it from the scratch: use the flag: `module.use_saved_model=true` and paste the path for the `checkpoint` in the config file.
