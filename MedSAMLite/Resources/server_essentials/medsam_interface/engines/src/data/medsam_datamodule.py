import copy
from typing import Any, Optional, Tuple

import albumentations as A
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class MedSAMDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_val_test_split: Tuple[int, int, int] = (0.9, 0.05, 0.05),
        batch_size: int = 16,
        num_workers: int = 16,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.dataset = dataset
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_dataset = self.dataset(num_workers=self.hparams.num_workers)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=train_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            val_dataset = copy.deepcopy(train_dataset)
            val_dataset.aug_transform = A.NoOp()
            self.data_val.dataset = self.data_test.dataset = val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = MedSAMDataModule(None)
