from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl

# To use the existing loader, we can import it.
# Assuming torch_gans.data.datasets.load_mnist_dataset
from .datasets import load_mnist_dataset # Relative import for within the same package


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.2, # Percentage of training data to use for validation
        normalize_mean: Tuple[float, ...] = (0.5,), # MNIST is single channel
        normalize_std: Tuple[float, ...] = (0.5,),  # To scale to [-1, 1] for Tanh
        seed: int = 42, # For reproducible splits
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.seed = seed

        # Define transforms
        transform_list = [transforms.ToTensor()]
        if self.normalize_mean and self.normalize_std:
            transform_list.append(transforms.Normalize(self.normalize_mean, self.normalize_std))
        self.transform = transforms.Compose(transform_list)

        # self.dims is required by Lightning an can be accessed through self.size()
        # Example: self.dims = (1, 28, 28) for MNIST
        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None


    @property
    def image_shape(self) -> Tuple[int, int, int]:
        # Standard MNIST shape
        return (1, 28, 28)

    def prepare_data(self):
        # Download data if not already present
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            if self.val_split > 0:
                train_size = int((1 - self.val_split) * len(mnist_full))
                val_size = len(mnist_full) - train_size
                self.data_train, self.data_val = random_split(
                    mnist_full, [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed)
                )
            else:
                self.data_train = mnist_full
                self.data_val = None # Or a small subset for basic validation if preferred

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        if not self.data_train:
            raise ValueError("Training data not setup. Call setup('fit') first.")
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True, # Usually good for GPU training
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.data_val:
            # If no validation split, can return None or a DataLoader with a small subset of test data
            return None
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if not self.data_test:
            raise ValueError("Test data not setup. Call setup('test') first.")
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
