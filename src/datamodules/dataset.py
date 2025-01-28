import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torchvision import datasets, transforms
import torch
import os



class COASTALDataModule (pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.save_hyperparameters()

        self.slurm_env = SLURMEnvironment()

        # Configurations
        self.batch_size  = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.data_dir    = config["data_dir"]

        # mean/std calculated from training data, with method: calc_norm_values.py
        mean_224 = (0.48039714, 0.476299, 0.44534707)
        std_224 = (0.16886398, 0.15295851, 0.16063267)

        mean_299 = (0.48039493, 0.47629836, 0.4453485)
        std_299 = (0.1743606, 0.15852651, 0.16594319)

        # Transformations for training data
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_224, std=std_224),
        ])

        # Transformations for validation data
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_224, std=std_224),
          ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_224, std=std_224),
          ])


    def setup(self, stage=None):

        # Print rank information
        local_rank = self.slurm_env.local_rank()
        global_rank = self.slurm_env.global_rank()
        print(f"local_rank={local_rank}, global_rank={global_rank}")

        """
        ImageFolder parameters:
          - root (str or pathlib.Path) – Root directory path.
          - transform (callable, optional) – A function/transform that takes in a PIL image and returns a transformed version
          - target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
          - loader (callable, optional) – A function to load an image given its path.
          - is_valid_file (callable, optional) – A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files)
          - allow_empty – If True, empty folders are considered to be valid classes. An error is raised on empty folders if False (default).
        """
        if stage == 'fit' or stage is None:
            self.train_dset = datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=self.train_transforms
            )
            self.val_dset = datasets.ImageFolder(
                root=f"{self.data_dir}/validation",
                transform=self.val_transforms
            )

        if stage == 'test':
            self.test_dset = datasets.ImageFolder(
                root=f"{self.data_dir}/test",
                transform=self.test_transforms  # Same transforms as validation
            )
            print(f"Total number of test samples: {len(self.test_dset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset            = self.train_dset,
            batch_size         = self.batch_size,
            num_workers        = self.num_workers,
            shuffle            = True,
            persistent_workers = True,
            pin_memory         = True,
            prefetch_factor    = 2,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset            = self.val_dset,
            batch_size         = self.batch_size,
            num_workers        = self.num_workers,
            shuffle            = False,
            persistent_workers = True,
            pin_memory         = True,
            prefetch_factor    = 2,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset            = self.test_dset,
            batch_size         = self.batch_size,
            num_workers        = self.num_workers,
            shuffle            = False,
            persistent_workers = True,
            pin_memory         = True,
            prefetch_factor    = 2,
        )