import os
import lightning.pytorch as pl
import hydra
import pyrootutils
import torch

from omegaconf import OmegaConf


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

@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="COASTresnet50")
def main(cfg):

    path = "/home/angelos.toutsios.gr/data/CS4321/HW1/teamsmt/out"

    checkpoint_path = f"{path}/2025-01-28_13-09-06/lightning_logs/version_0/checkpoints/epoch=9-val_loss=0.92259.ckpt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")


    print(f"Loading best model from {checkpoint_path}")
    # print("Loaded Config:\n", OmegaConf.to_yaml(cfg))
    model: pl.LightningModule = hydra.utils.instantiate(cfg.module, ckpt_path=checkpoint_path)
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    print("Running test evaluation...")
    datamodule.setup(stage="test")  # Ensure test dataset is initialized
    results = trainer.test(model, datamodule.test_dataloader())  # Capture test results
    print("Test Results:", results)


if __name__ == "__main__":
    main()