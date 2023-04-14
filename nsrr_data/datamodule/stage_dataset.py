from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from h5py import File
from joblib import Memory, delayed
from sklearn import preprocessing
from torch.utils.data import Dataset

from nsrr_data.datamodule.mixins import PlottingMixin, RecordDataset
from nsrr_data.utils.default_event_matching import match_events_localization_to_default_localizations
from nsrr_data.utils.h5_utils import load_waveforms
from nsrr_data.utils.logger import get_logger
from nsrr_data.utils.parallel_bar import ParallelExecutor

logger = get_logger()
SCALERS = {"robust": preprocessing.RobustScaler(), "standard": preprocessing.StandardScaler()}


@dataclass
class SleepStageDataset(Dataset):
    """

    Args:
        records (List[pathlib.Path])                : List of Path objects to .h5 files.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        fs (int)                                    : Sampling frequency, Hz.
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_records (int)                             : Total number of records to include (default None).
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        transform (Callable)                        : A Callable object to transform signal data by STFT, Morlet transforms or multitaper spectrograms.
                                                    : See the transforms/ directory for inspiration.
        sequence_length (int)                       : Number of 30 s epochs to include in sequence.

    """

    records: List[Path]
    sequence_length: int
    cache_data: bool = False
    fs: int = 128
    n_jobs: int = 1
    n_records: int = None
    picks: List[str] = None
    transform: Callable = None
    scaling: str = "robust"

    def __post_init__(self):
        self.cache_dir = Path("") / "data" / ".cache"
        self.n_records = len(self.records)
        self.n_channels = len(self.picks)
        if self.cache_data:
            print(f"Using cache for data prep: {self.cache_dir.resolve()}")
            memory = Memory(self.cache_dir.resolve(), mmap_mode="r", verbose=0)
            get_metadata = memory.cache(get_record_metadata)
        else:
            get_metadata = get_record_metadata
        self.record_metadata = {}
        self.index_to_record = []
        self.scalers = []

        if self.records is not None:
            # logger.info(f"Prefetching study metadata using {self.n_jobs} workers:")
            sorted_data = ParallelExecutor(n_jobs=-4, prefer="processes")(total=len(self.records))(
                delayed(get_metadata)(filename=record, sequence_length=self.sequence_length)
                for record in set(self.records)
            )
            logger.info("Prefetching finished")
        else:
            raise ValueError(f"Please specify a data directory, received: {self.records}")

        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"][1]]
        self.metadata = dict([s["metadata"] for s in sorted_data])
        self.stages = dict([s["stages"] for s in sorted_data])

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):

        record = self.index_to_record[idx]["record"]
        window_index = self.index_to_record[idx]["window_idx"]
        # window_start =   # self.index_to_record[idx]["window_start"]

        # Load specific channels and location
        signal = load_waveforms(self.metadata[record]["filename"], self.picks, window=window_index)

        # Get valid stages
        stages = self.stages[record][::30][window_index]

        # Optionally transform the signal
        if self.transform is not None:
            signal = self.transform(signal)

        return {
            "signal": signal,
            "stages": np.array(stages),
            "record": f"{record}_{window_index.start:04d}-{window_index.stop-1:04d}",
        }


def get_record_metadata(filename: str, sequence_length: int):

    # Get signal metadata
    with File(filename, "r") as h5:

        # Get the waveforms and shape info
        N, C, T = h5["data"]["scaled"].shape
        stages = h5["stages"][:]

        # Set metadata
        index_to_record = [
            {"record": filename.stem, "window_idx": slice(x, x + sequence_length)}
            for x in range(N - sequence_length + 1)
        ]
        metadata = {"n_channels": C, "length": T, "filename": filename}

    return dict(
        index_to_record=(filename.stem, index_to_record),
        metadata=(filename.stem, metadata),
        stages=(filename.stem, stages),
    )
