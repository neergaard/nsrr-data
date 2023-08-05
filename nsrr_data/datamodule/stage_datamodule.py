from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nsrr_data.datamodule.stage_dataset import SleepStageDataset
from nsrr_data.utils.collate_fn import collate
from nsrr_data.utils.partitioning import get_train_validation_test


@dataclass
class SleepStageDataModule(LightningDataModule):
    """SleepStageDataModule containing logic to contain and split a dataset.
    It also contains methods to return PyTorch DataLoaders for each split.

    Args:
        data_dir (str)                              : Directory to .h5 data files
        batch_size (int)                            : Number of data windows to include in a batch.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        fs (int)                                    : Sampling frequency, Hz.
        n_eval (int)                                : Number of validation subjects to include
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_test (int)                                : Number of test subjects to include
        n_records (int)                             : Total number of records to include (default None).
        num_workers (int)                           : Number of workers to use for dataloaders.
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        seed (int)                                  : Random seed
        sequence_length (int)                       : Number of 30 s epochs to include in sequence.

    """

    # Partition specific
    data_dir: str
    n_test: int = 1000
    n_eval: int = 200
    seed: int = 1337
    overfit: bool = False

    # Dataset specific
    sequence_length: int = None
    cache_data: bool = False
    fs: int = None
    n_jobs: int = None
    n_records: int = None
    picks: Optional[List[str]] = None
    transform: Callable = None
    scaling: str = None

    # Dataloader specific
    batch_size: int = 1
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).resolve()
        partitions = get_train_validation_test(
            self.data_dir,
            number_test=self.n_test,
            number_validation=self.n_eval,
            seed=self.seed,
            n_records=self.n_records,
        )
        self.train_records = partitions["train"]
        self.eval_records = partitions["eval"] if not self.overfit else partitions["train"]
        self.test_records = partitions["test"]
        self.n_channels = len(self.picks)
        self.example_input_array = torch.randn(self.batch_size, self.n_channels, self.sequence_length)

        self.dataset_kwargs = dict(
            sequence_length=self.sequence_length,
            cache_data=self.cache_data,
            fs=self.fs,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            picks=self.picks,
            transform=self.transform,
            scaling=self.scaling,
        )
        self.save_hyperparameters(
            ignore=["train_records", "eval_records", "test_records", "example_input_array", "dataset_kwargs"]
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        self.train = SleepStageDataset(self.train_records, **self.dataset_kwargs)
        self.eval = SleepStageDataset(self.eval_records, **self.dataset_kwargs)
        self.test = SleepStageDataset(self.test_records, **self.dataset_kwargs)
        self.output_dims = self.example_input_array.numpy().shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = SleepStageDataModule("data/mros/processed")
    print(repr(dm))
    print(dm)
