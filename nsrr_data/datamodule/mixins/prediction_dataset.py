import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class RecordDataset:
    """Return signal data from a specific record as a batch of continuous
    windows. Overlap in seconds allows overlapping among windows in the
    batch. The last data points will be ignored if their length is
    inferior to window_size.
    """

    def get_record_dataset(self, record, batch_size, stride=None):
        class PredictionDataset(Dataset):
            def __init__(
                self, record, batch_size, window, fs, window_size, channels_start, channels_stop, stride=None
            ):
                super().__init__()
                self.record = record
                self.channels_start = channels_start
                self.channels_stop = channels_stop
                self.signal_size = record["size"]
                self.window_size = window_size

                self.stride = int((stride if stride is not None else window) * fs)
                self.t = np.arange(self.signal_size, dtype=np.int32)
                self.n_windows = int((self.signal_size - self.stride) / (self.window_size - self.stride))

            def __len__(self):
                return self.n_windows

            def __getitem__(self, idx):

                start = idx * self.stride
                stop = start + self.window_size

                return (
                    torch.from_numpy(self.record["data"][self.channels_start : self.channels_stop, start:stop]),
                    self.t[start:stop],
                )

            @staticmethod
            def collate_fn(batch):
                x = torch.stack([b[0] for b in batch])
                t = np.stack([b[1] for b in batch])
                return x, t

        prediction_dataset = PredictionDataset(
            self.signals[record],
            batch_size,
            self.window,
            self.fs,
            self.window_size,
            self.channels_start,
            self.channels_end,
            stride,
        )
        return DataLoader(
            prediction_dataset,
            batch_size=64,
            collate_fn=PredictionDataset.collate_fn,
            num_workers=12,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
