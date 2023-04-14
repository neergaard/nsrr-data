from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from librosa import power_to_db, stft
from librosa.feature import melspectrogram
from librosa.display import specshow
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nsrr_data.utils.logger import get_logger
from nsrr_data.utils.plotting import plot_data, plot_spectrogram

logger = get_logger()


class PlottingMixin:
    def plot_signals(self, idx: int, channel_names: List[str] = None, ax: Optional[Axes] = None):

        # Temporarily remove transforms
        transform = getattr(self, "transform", None)
        self.transform = None

        sample = self[idx]
        record = sample["record"]
        data = sample["signal"]
        events = sample["events"]
        fs = self.fs
        if channel_names is None:
            channel_names = self.picks

        fig, ax = plot_data(
            data, events, fs, channel_names=channel_names, title=f"{record} | No. events: {len(events)}", ax=ax
        )

        # Re-attach transform
        self.transform = transform

    def plot_spect(
        self,
        idx: int,
        channel_idx: int = 0,
        display_type: str = "hz",
        step_size: Optional[int] = None,
        window_size: Optional[int] = None,
        nfft: Optional[int] = None,
        ax: Optional[Axes] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> Tuple[Figure, Axes]:

        # Check for any transforms
        transform = getattr(self, "transform", None)

        sample = self[idx]
        record = sample["record"]
        data = sample["signal"]
        events = sample["events"]
        fs = self.fs

        if transform is None:
            S = stft(data[channel_idx], n_fft=nfft, hop_length=step_size, win_length=window_size)
            S_db = power_to_db(np.abs(S) ** 2, top_db=50) / 50
        else:
            S_db = data[channel_idx]
            step_size = transform.step_size
            window_size = transform.segment_size
            nfft = transform.nfft

        # logger.info(f"{S_db.shape=}")
        if display_type == "mel":
            S_db = melspectrogram(S=S_db, sr=fs, n_fft=nfft, hop_length=step_size, win_length=window_size, n_mels=20)
        # logger.info(f"{S_db.shape=}")
        plot_spectrogram(
            S_db,
            fs=fs,
            step_size=step_size,
            window_length=window_size,
            nfft=nfft,
            ax=ax,
            display_type=display_type,
            fmin=fmin,
            fmax=fmax,
        )
        # specshow(S_db, sr=fs, hop_length=step_size, n_fft=nfft, win_length=window_size, y_axis="hz", x_axis="time")

    def plot(
        self,
        idx: int,
        channel_idx: int = 0,
        display_type: str = "hz",
        channel_names: List[str] = None,
        step_size: Optional[int] = None,
        window_size: Optional[int] = None,
        nfft: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> None:

        fig, ax = plt.subplots(nrows=2, figsize=(25, 8), sharex=True, gridspec_kw={"hspace": 0.01})
        self.plot_signals(idx=idx, channel_names=channel_names, ax=ax[0])
        self.plot_spect(
            idx=idx,
            channel_idx=channel_idx,
            step_size=step_size,
            window_size=window_size,
            nfft=nfft,
            ax=ax[1],
            display_type=display_type,
            fmin=fmin,
            fmax=fmax,
        )
