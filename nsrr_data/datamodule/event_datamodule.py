from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nsrr_data.datamodule.event_dataset import SleepEventDataset
from nsrr_data.utils.collate_fn import collate
from nsrr_data.utils.partitioning import get_train_validation_test
from nsrr_data.utils.default_event_matching import get_overlapping_default_events


@dataclass
class SleepEventDataModule(LightningDataModule):
    """SleepEventDataModule containing logic to contain and split a dataset.
    It also contains methods to return PyTorch DataLoaders for each split.

    Args:
        data_dir (str)                              : Directory to .h5 data files
        batch_size (int)                            : Number of data windows to include in a batch.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        default_event_window_duration (List[int])   : List of default event window durations. Not used for DETR (default 10).
        event_buffer_duration (int)                 : Small buffer for window sampling in seconds (default 1).
        events (Dict[str, str])                     : Dictionary containing event codes as keys and event names as values.
                                                      Eg. {'ar': 'Arousal', 'lm': 'Leg movement', 'sdb': 'Sleep-disordered breathing'}
        factor_overlap (int)                        : Overlap between successive default event windows. Not used for DETR (default 2)
        fs (int)                                    : Sampling frequency, Hz.
        matching_overlap (float)                     : Threshold for matching events for default event windows (default 0.7)
        n_eval (int)                                : Number of validation subjects to include
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_test (int)                                : Number of test subjects to include
        n_records (int)                             : Total number of records to include (default None).
        num_workers (int)                           : Number of workers to use for dataloaders.
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        seed (int)                                  : Random seed
        transform (Callable)                        : A Callable object to transform signal data by STFT, Morlet transforms or multitaper spectrograms.
                                                    : See the transforms/ directory for inspiration.
        window_duration (int)                       : Duration of data segment in seconds.

    """

    # Partition specific
    data_dir: str
    n_test: int = 1000
    n_eval: int = 200
    seed: int = 1337
    overfit: bool = False

    # Dataset specific
    events: dict = None
    window_duration: int = None
    cache_data: bool = False
    default_event_window_duration: List[int] = field(default_factory=lambda: ([3, 15, 30]))
    event_buffer_duration: int = 1
    factor_overlap: int = 2
    fs: int = None
    matching_overlap: float = None
    n_jobs: int = None
    n_records: int = None
    picks: list = None
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
        self.n_classes = len(self.events.keys())
        self.n_channels = len(self.picks)
        self.window_size = self.window_duration * self.fs
        self.example_input_array = torch.randn(self.batch_size, self.n_channels, self.window_size)
        self.localizations_default = get_overlapping_default_events(
            window_size=self.window_size,
            default_event_sizes=[d * self.fs for d in self.default_event_window_duration],
            factor_overlap=self.factor_overlap,
        )

        self.dataset_kwargs = dict(
            events=self.events,
            window_duration=self.window_duration,
            cache_data=self.cache_data,
            default_event_window_duration=self.default_event_window_duration,
            event_buffer_duration=self.event_buffer_duration,
            fs=self.fs,
            localizations_default=self.localizations_default,
            matching_overlap=self.matching_overlap,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            picks=self.picks,
            transform=self.transform,
            scaling=self.scaling,
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit":
            self.train = SleepEventDataset(self.train_records, **self.dataset_kwargs)
            self.eval = SleepEventDataset(self.eval_records, **self.dataset_kwargs)
            self.output_dims = [self.batch_size] + self.train.output_dims
        elif stage == "test":
            self.test = SleepEventDataset(self.eval_records, **self.dataset_kwargs)
            self.output_dims = [self.batch_size] + self.test.output_dims

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
    dm = SleepEventDataModule("data/mros/processed")
    print(repr(dm))
    print(dm)
