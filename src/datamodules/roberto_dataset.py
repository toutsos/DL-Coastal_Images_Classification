import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import models
import torch
import os



class GenericDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.save_hyperparameters()

        self.slurm_env = SLURMEnvironment()

        # Configurations
        self.batch_size  = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.data_dir    = config["data_dir"]
        self.rotation    = config["rotation"]

        # mean/std calculated from training data, with method: calc_norm_values.py
        mean_224 = (0.485, 0.456, 0.406)
        std_224 = (0.229, 0.224, 0.225)

        # Transformations for training data
        if self.rotation == 0:
            self.train_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_224, std=std_224),
            ])

        else:
            self.train_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=self.rotation, expand=True),
                transforms.CenterCrop(224),  # Crop back to 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_224, std=std_224),
            ])

        # Transformations for validation data
        self.val_transforms = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_224, std=std_224),
          ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((244, 244)),
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
          - root (str or pathlib.Path) - Root directory path.
          - transform (callable, optional) - A function/transform that takes in a PIL image and returns a transformed version
          - target_transform (callable, optional) - A function/transform that takes in the target and transforms it.
          - loader (callable, optional) - A function to load an image given its path.
          - is_valid_file (callable, optional) - A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files)
          - allow_empty - If True, empty folders are considered to be valid classes. An error is raised on empty folders if False (default).
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
            print(f"Looking for test dataset in: {self.data_dir}/test")
            if not os.path.exists(f"{self.data_dir}/test"):
                raise FileNotFoundError(f"Test dataset not found in: {self.data_dir}/test")

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

class ViTDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.save_hyperparameters()

        self.slurm_env = SLURMEnvironment()

        # Configurations
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.data_dir = config["data_dir"]

        # transformations
        self.rotation = config["rotation"]

        # Load standard ViT-B/16 inference transforms
        vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        vit_mean = [0.485, 0.456, 0.406]
        vit_std = [0.229, 0.224, 0.225]

        self.inference_transforms = vit_weights.transforms()

        # Explicitly define the transforms from ViT spec
        self.train_transforms = transforms.Compose([
            transforms.Resize(size=256, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(degrees=20),  # Random rotation between -15 and 15 degrees
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Random crop
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),
            transforms.Normalize(mean=vit_mean, std=vit_std),
        ])

        # Validation & Test Transforms (No random augmentations)
        self.val_transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=vit_mean, std=vit_std),
        ])

        self.test_transforms = self.val_transforms  # Same as validation

    def setup(self, stage=None):
        # Print rank information
        local_rank = self.slurm_env.local_rank()
        global_rank = self.slurm_env.global_rank()
        print(f"local_rank={local_rank}, global_rank={global_rank}")

        if stage == 'fit' or stage is None:
            self.train_dset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, "train"),
                transform=self.train_transforms
            )
            self.val_dset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, "validation"),
                transform=self.val_transforms
            )

        if stage == 'test':
            print(f"Looking for test dataset in: {self.data_dir}/test")
            if not os.path.exists(f"{self.data_dir}/test"):
                raise FileNotFoundError(f"Test dataset not found in: {self.data_dir}/test")

            self.test_dset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, "test"),
                transform=self.test_transforms
            )
            print(f"Total number of test samples: {len(self.test_dset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )
