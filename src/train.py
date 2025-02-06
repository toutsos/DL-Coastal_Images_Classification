import lightning.pytorch as pl
import pyrootutils
import torch
import hydra
import os
import sys
from pathlib import Path

# Calculate the project root (one level up from the src directory)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.modules.ResNet50Model import COASTALResNet50

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["configs"],
    pythonpath=True,
    dotenv=True,
)

torch.set_float32_matmul_precision(
    "medium"
) # for better performance on multi-GPU training

# Set to make webdataset .lock file readable by other users
os.umask(0)

def train(cfg):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    if cfg.module.use_saved_model:
      if cfg.module.load_model_path:
        # Use a pretrained model
        print('-'*50)
        print(f"Loading model from checkpoint: {cfg.module.load_model_path}")
        # model: pl.LightningModule = COASTALResNet50.load_from_checkpoint(use_pretrained, fine_tune, frozenlayers, lr, cfg.module.load_model_path)
        print(f"Model is trained with the next parameters:")
        print(f"Pretrain = {cfg.module.use_pretrained}")
        print(f"Fine Tune = {cfg.module.fine_tune}")
        print(f"Frozen Layers = {cfg.module.frozen_layers}")
        print(f"Learning Rate = {cfg.module.lr}")
        print(f"Weight Decay = {cfg.module.weight_decay}")
        print('-'*50)

        model = COASTALResNet50.load_from_checkpoint(
          cfg.module.load_model_path,
          # use_pretrained=cfg.module.use_pretrained,  # Override specific params
          weight_decay = cfg.module.weight_decay,
          fine_tune=cfg.module.fine_tune,
          frozen_layers=cfg.module.frozen_layers,
          lr=cfg.module.lr  # Example: Change learning rate
        )

      else:
        print('-'*50)
        print("No checkpoint path provided, training from scratch.")
        model: pl.LightningModule = hydra.utils.instantiate(cfg.module)
        print('-'*50)
    else:
      # No checkpoint path provided, train from scratch
      print('-'*50)
      print("No checkpoint provided, training from scratch.")
      model: pl.LightningModule = hydra.utils.instantiate(cfg.module)
      print('-'*50)

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    # print("------------------------------------------------------------------------------------------------------")
    # for name, param in model.named_parameters():
    #   print(f"{name}: {param.shape}, requires_grad={param.requires_grad}\n")
    #   print(f"First 5 values: {param.view(-1)[:5]}")  # Print only the first 5 values
    # print("------------------------------------------------------------------------------------------------------")


    fit_kwargs = {
        "model": model,
        "datamodule": datamodule
    }

    try:
        # This method RESUMES the training from this checkpoint
        fit_kwargs['ckpt_path'] = cfg.ckpt_path
        print('-'*50)
        print(f'Model is trained from checkpoint: {fit_kwargs["ckpt_path"]}')
        print('-'*50)
    except Exception as e:
        pass

    print("------------------------------------------------------------------------------------------------------")
    print("Model Hyperparameters:\n", model.hparams)
    print("------------------------------------------------------------------------------------------------------")
    trainer.fit(**fit_kwargs)

    print("Starting test phase...")
    trainer.test(**fit_kwargs)  # Ensure test phase is executed
    print("Test phase completed.")

@hydra.main(version_base="1.3", config_path=str(root / "configs"))
def main(cfg):
    pl.seed_everything(42) # for reproducibility
    train(cfg)

if __name__ == "__main__":
    main()
