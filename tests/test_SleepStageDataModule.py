from nsrr_data.datamodule.stage_datamodule import SleepStageDataModule


config = dict(
    data_dir="data/processed/mros/ar",
    n_test=10,
    n_eval=10,
    seed=1337,
    sequence_length=10,
    cache_data=True,
    fs=128,
    n_jobs=-1,
    n_records=30,
    picks=["C3", "C4"],
    transform=None,
    scaling="robust",
    batch_size=8,
    num_workers=0,
)


class TestClass_SleepStageDataModule:

    config = config
    dm = SleepStageDataModule(**config)

    def test_instance(self):
        print(self.dm)

        self.dm.setup()

    def test_train_iterator(self):
        self.dm.setup()
        train = self.dm.train

        for idx, batch in enumerate(train):
            print(f"{idx} | Data.shape: {batch['signal'].shape}")
            break
