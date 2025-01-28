import lightning.pytorch as pl
import pyrootutils
import torch
import hydra
import os

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
    model: pl.LightningModule = hydra.utils.instantiate(cfg.module)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    fit_kwargs = {
        "model": model,
        "datamodule": datamodule
    }

    try:
        fit_kwargs['ckpt_path'] = cfg.ckpt_path
    except Exception:
        pass

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
