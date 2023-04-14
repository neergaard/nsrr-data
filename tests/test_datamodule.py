from tqdm import tqdm

from nsrr_data.datamodule.event_datamodule import SleepEventDataModule
from nsrr_data.datamodule.transforms import MorletTransform, MultitaperTransform, STFTTransform

print("Setting up test")
params = dict(
    data_dir="data/processed/mros/lm",
    batch_size=16,
    n_eval=2,
    n_test=2,
    num_workers=4,
    seed=1337,
    # events={"ar": "Arousal", "lm": "Leg movement", "sdb": "Sleep-disordered breathing"},
    # events={"ar": "Arousal"},
    events={"lm": "Leg movement"},
    window_duration=600,  # seconds
    cache_data=True,
    default_event_window_duration=[3],  # [10] for LM, [30] for SDB
    event_buffer_duration=3,
    factor_overlap=2,
    fs=128,  # 64 for both LM and SDB
    matching_overlap=0.5,
    n_jobs=-1,
    n_records=6,
    # picks=["c3", "c4", "eogl", "eogr", "chin", "legl", "legr", "nasal", "abdo", "thor"],
    # picks=["c3", "c4", "eogl", "eogr", "chin"],
    picks=["legl", "legr"],
    # transform=MultitaperTransform(128, 0.5, 35.0, tw=8.0, normalize=True),
    transform=STFTTransform(
        fs=128, segment_size=int(4.0 * 128), step_size=int(0.125 * 128), nfft=1024, normalize=True
    ),  # Change 128 to matching sampling frequency
    scaling="robust",
)


class TestDataModule:

    print("Creating DataModule")
    try:
        dm = SleepEventDataModule(**params)
        print(dm)
    except:
        dm = None

    def test_instance(self):

        assert isinstance(self.dm, SleepEventDataModule)

    def test_setup_fit(self):

        print("Setting up partitions")
        self.dm.setup("fit")
        # assert self.dm.has_setup_fit
        assert hasattr(self.dm, "train")
        assert hasattr(self.dm, "eval")
        print(f"Train len: {len(self.dm.train)}")
        print(f"Eval len: {len(self.dm.eval)}")

    def test_datashape(self):

        print("Setting up partitions")
        self.dm.setup("fit")

        print("Getting next item")
        signals, events, records, clf, loc, ss = next(iter(self.dm.train_dataloader()))

        assert signals.ndim == len(self.dm.output_dims)  # (Batch size, number of channels, length of segments)
        assert list(signals.shape) == self.dm.output_dims

    def test_passthrough_train(self):

        print("Setting up partitions")
        self.dm.setup("fit")

        all_pass = False
        dl = self.dm.train_dataloader()
        for batch in tqdm(dl):
            pass
