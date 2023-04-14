from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from joblib import Memory, delayed
from sklearn import preprocessing
from torch.utils.data import Dataset

from nsrr_data.datamodule.mixins import PlottingMixin, RecordDataset
from nsrr_data.utils.default_event_matching import match_events_localization_to_default_localizations
from nsrr_data.utils.h5_utils import get_record_metadata, load_waveforms
from nsrr_data.utils.logger import get_logger
from nsrr_data.utils.parallel_bar import ParallelExecutor

logger = get_logger()
SCALERS = {"robust": preprocessing.RobustScaler(), "standard": preprocessing.StandardScaler()}


@dataclass
class SleepEventDataset(RecordDataset, PlottingMixin, Dataset):
    """

    Args:
        records (List[pathlib.Path])                : List of Path objects to .h5 files.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        class_probabilities (Dict[str, float])       : Dictionary containing custom class probabilities (default None).
        default_event_window_duration (List[int])   : List of default event window durations. Not used for DETR (default [3, 15, 30]).
        event_buffer_duration (int)                 : Small buffer for window sampling in seconds (default 3).
        events (Dict[str, str])                     : Dictionary containing event codes as keys and event names as values.
                                                      Eg. {'ar': 'Arousal', 'lm': 'Leg movement', 'sdb': 'Sleep-disordered breathing'}
        factor_overlap (int)                        : Overlap between successive default event windows. Not used for DETR (default 2)
        fs (int)                                    : Sampling frequency, Hz.
        localizations_default (np.ndarray)          : Numpy array containing default event windows. Can be calculated using the get_overlapping_default_events() function.
        matching_overlap (float)                     : Threshold for matching events for default event windows (default 0.7)
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_records (int)                             : Total number of records to include (default None).
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        transform (Callable)                        : A Callable object to transform signal data by STFT, Morlet transforms or multitaper spectrograms.
                                                    : See the transforms/ directory for inspiration.
        window_duration (int)                       : Duration of data segment in seconds.

    """

    records: List[Path]
    events: Optional[Dict] = field(
        default_factory=lambda: ({"ar": "Arousal", "lm": "Leg movement", "sdb": "Sleep-disordered breathing"})
    )
    window_duration: int = 20 * 60
    cache_data: bool = False
    class_probabilities: dict = None
    default_event_window_duration: List[int] = field(default_factory=lambda: ([3, 15, 30]))
    event_buffer_duration: float = 3
    factor_overlap: int = 2
    fs: int = 128
    localizations_default: np.ndarray = None
    matching_overlap: float = 0.5
    n_jobs: int = 1
    n_records: int = None
    picks: List[str] = None
    transform: Callable = None
    scaling: str = "robust"

    def __post_init__(self):
        self.cache_dir = Path("") / "data" / ".cache"
        # self.records = sorted(self.data_dir.glob("*.h5"))[: self.n_records]
        self.n_records = len(self.records)
        self.n_channels = len(self.picks)
        if self.events is not None:
            self.n_classes = len(self.events)
            self.event_names = [ev for ev in self.events.values()]
        if self.cache_data:
            # logger.info(f'Using cache for data prep: {self.cache_dir.resolve()}')
            print(f"Using cache for data prep: {self.cache_dir.resolve()}")
            memory = Memory(self.cache_dir.resolve(), mmap_mode="r", verbose=0)
            get_metadata = memory.cache(get_record_metadata)
        else:
            get_metadata = get_record_metadata
        self.event_buffer = self.event_buffer_duration * self.fs
        self.default_event_window_size = [
            int(default_window * self.fs) for default_window in self.default_event_window_duration
        ]
        self.window_size = int(self.window_duration * self.fs)
        self.record_metadata = {}
        self.index_to_record = []
        self.index_to_record_event = []
        self.scalers = []

        if self.records is not None:
            # logger.info(f"Prefetching study metadata using {self.n_jobs} workers:")
            # with parallel_backend("loky", inner_max_num_threads=2):
            sorted_data = ParallelExecutor(n_jobs=-4, prefer="processes")(total=len(self.records))(
                delayed(get_metadata)(
                    filename=record,
                    events=self.events,
                    fs=self.fs,
                    window_size=self.window_size,
                )
                for record in set(self.records)
            )
            # logger.info("Prefetching finished")
        else:
            raise ValueError(f"Please specify a data directory, received: {self.records}")

        self.event_data = dict([s["event_data"] for s in sorted_data])
        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"][1]]
        self.index_to_record_event = [sub for s in sorted_data for sub in s["index_to_record_event"][1]]
        self.n_events = dict([s["n_events"] for s in sorted_data])
        self.metadata = dict([s["metadata"] for s in sorted_data])
        self.stages = dict([s["stages"] for s in sorted_data])

        # Set the class probabilities
        if not self.class_probabilities:
            self.class_probabilities = {k: 1 / self.n_classes for k in self.events.keys()}
        else:
            # Remember to normalize class probabilities
            self.class_probabilities = {
                k: v / sum(self.class_probabilities.values()) for k, v in self.class_probabilities.items()
            }

        self.matching = partial(
            match_events_localization_to_default_localizations,
            self,
            _localizations_default=self.localizations_default,
            threshold_overlap=self.matching_overlap,
            window=self.window_duration,
        )

        if self.transform is None:
            self.output_dims = [self.n_channels, self.window_size]
        else:
            self.output_dims = self.transform.calculate_output_dims(self.window_size)

    def __len__(self):
        # return len(self.index_to_record_event)
        return len(self.index_to_record)

    def __getitem__(self, idx):

        record = self.index_to_record[idx]["record"]
        window_index = self.index_to_record[idx]["idx"]
        window_start = self.index_to_record[idx]["window_start"]

        # Load specific channels and location
        signal = load_waveforms(self.metadata[record]["filename"], self.picks, window=window_index)

        # Get only event data inside window
        events_data = []
        for event_name, event in self.event_data[record].items():
            starts, stops = event["data"][:, 0], np.sum(event["data"], axis=1)
            valid_events = [
                (start < window_start + self.window_size) and (stop > window_start)
                for (start, stop) in zip(starts, stops)
            ]
            for valid_start, valid_stop in zip(starts[valid_events], stops[valid_events]):
                events_data.append(
                    (
                        valid_start,
                        valid_stop,
                        event["label"] - event["label"],  # This is to ensure that the event label id is 0
                    )
                )  # Maybe add the non-event class?

        # We normalize wrt. to the current index and window size
        try:
            events = np.array(events_data).astype(np.float32)
            events[:, :2] = (events[:, :2] - window_start) / self.window_size
        except IndexError:
            events = np.array([[], [], []]).T

        # Get valid stages
        stages = self.stages[record][window_start // self.fs : window_start // self.fs + self.window_duration]

        # Match the associated events with default event windows
        if events is not None:
            (localizations_target, classifications_target) = self.matching(events=events)
            localizations_target = localizations_target.squeeze(0)
            classifications_target = classifications_target.squeeze(0)
        else:
            (localizations_target, classifications_target) = (np.array([[], []]).T, np.array([]))

        # Optionally transform the signal
        if self.transform is not None:
            signal = self.transform(signal, events)

        return {
            "signal": signal,
            "events": events,
            "stages": np.array(stages),
            # "record": self.index_to_record_event[idx]["record"],
            "record": f"{record}_{window_index:04d}",
            "localizations_target": localizations_target,
            "classifications_target": classifications_target,
        }

    def extract_balanced_multiclass_data(self, record, index=None):
        """Extract balanced data in a multi-class problem. This function samples the events and shifts the extracted
        window based on a specific sample.
        """
        num_events_class = self.n_events[record]
        class_probs = self.class_probabilities
        choice = None
        while choice is None:
            choice = np.random.choice([k for k in class_probs.keys()], p=[v for v in class_probs.values()])
            if num_events_class[choice] == 0:
                choice = None
        if choice is not None:
            class_events = self.event_data[record][choice]["data"]
            if class_events.shape[0] == 0:
                window_start = np.random.randint(self.signals[record]["length"] - self.window_size)
            else:
                random_event_idx = np.random.randint(class_events.shape[0])
                event_start, event_stop = class_events[random_event_idx, 0], class_events[random_event_idx].sum()
                window_start = np.random.randint(
                    event_stop + self.event_buffer - self.window_size,
                    event_start - self.event_buffer,
                )
                window_start = np.clip(
                    window_start,
                    0,
                    self.metadata[record]["length"] - self.window_size,
                )
            psg_data, events_data = self.get_sample(record, window_start)
        else:
            psg_data, events_data = self.get_sample(record)
        return psg_data, events_data

    def get_sample(self, record, index):
        """Return a sample [sata, events] from a record at a particular index"""
        # with h5py.File(f"{self.h5_directory}/{record}", 'r') as h5:
        #     signal_data = h5['data'][self.channels_start:self.channels_end, index : index + self.window_size]
        signal_data = load_waveforms(
            self.metadata[record]["filename"], self.picks, window=slice(index, index + self.window_size)
        )  # [:, index : index + self.window_size]
        events_data = []
        for event_name, event in self.event_data[record].items():
            starts, stops = event["data"][:, 0], np.sum(event["data"], axis=1)
            valid_events = [
                (start < index + self.window_size) and (stop > index) for (start, stop) in zip(starts, stops)
            ]
            for valid_start, valid_stop in zip(starts[valid_events], stops[valid_events]):
                events_data.append(
                    (
                        valid_start,
                        valid_stop,
                        event["label"],
                    )
                )

        # We normalize wrt. to the current index and window size
        events_data = np.array(events_data).astype(np.float32)
        events_data[:, :2] = (events_data[:, :2] - index) / self.window_size
        # TODO: I don't think this is the correct way of doing. It should be irrespective of the stopping inside the window, because otherwise we drop some events.
        # We should just look at if any events are **starting** OR **stopping** in the window and take these with us further.
        # starts, durations = event["data"][:, 0], event["data"][:, 1]
        # starts_relative = (starts - index) / self.window_size
        # durations_relative = durations / self.window_size
        # stops_relative = starts_relative + durations_relative
        # valid_starts_index = np.where((starts_relative > 0) * (starts_relative < 1))[0]
        # valid_stops_index = np.where((stops_relative > 0) * (stops_relative < 1))[0]
        # valid_indexes = set(list(valid_starts_index) + list(valid_stops_index))
        # for valid_index in valid_indexes:
        #     if valid_index in valid_starts_index:
        #         if valid_index in valid_stops_index:
        #             events_data.append(
        #                 (float(starts_relative[valid_index]), float(stops_relative[valid_index]), event["label"],)
        #             )
        #     else:
        #         if valid_index in valid_starts_index:
        #             pass
        #         if (1 - starts_relative[valid_index]) / durations_relative[valid_index] > self.minimum_overlap:
        #             events_data.append((float(starts_relative[valid_index]), 1, event["label"]))
        #         elif valid_index in valid_stops_index:
        #             if stops_relative[valid_index] / durations_relative[valid_index] > self.minimum_overlap:
        #                 events_data.append((0, float(stops_relative[valid_index]), event["label"]))

        return signal_data, events_data


if __name__ == "__main__":
    ds = SleepEventDataset()
    logger.info(ds)
    logger.info(repr(ds))
    for batch in ds:
        print(
            f"Record: {batch['record']} | Signal shape: {batch['signal'].shape} | Unique events: {list(np.unique(batch['events'][:, -1]).astype(int))} | No. events: {list(np.bincount(batch['events'][:, -1].astype(int)))}"
        )
        # print(batch)
