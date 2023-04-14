# NSRR Data

NOTE: This README.md is currently under revision!

This repository contains code for downloading and preprocessing data from the [National Sleep Research Resource](https://sleepdata.org).

Currently supported datasets:
- MrOS
- SHHS
- WSC

## Table of contents

1. [Table of contents](#table-of-contents)
2. [Requirements](#requirements)
3. [Installation](#installation)
   1. [Ruby configuration](#ruby-configuration)
4. [Usage](#usage)
5. [Data preprocessing](#data-preprocessing)

## Requirements
```
- python==3.10
- librosa==0.9.2
- mne==1.2.1
- numpy==1.23.3
- scikit-learn==1.1.1
- pytorch==1.13.0
- pytorch-lightning==1.7.7
- rich==12.6.0
- ipympl
- pytest (for writing unit tests)
```

## Installation
The data pipeline and datamodule can be installed into your virtual environment by `cd`'ing into project folder and running `pip`:
```bash
git clone https://www.github.com/neergaard/nsrr-data.git
cd nsrr-data
pip install -e .
```
This will install an editable (`-e`) version of the `nsrr-data` package (meaning that you can edit the code and it will update the install automatically).

### Ruby configuration
To install the NSRR Ruby Gem run:
```bash
gem install nsrr --no-document
```
There's a bug in which the Ruby gems cannot be found by the executable.
To fix, run
```bash
ln -s $CONDA_PREFIX/bin/ruby $CONDA_PREFIX/share/rubygems/bin/ruby
```

## Usage
The dataset and associated data module can be found in the `nsrr_data/datamodule/` directory.
The `SleepEventDataset` class contains logic to load and iterate over individual segments for each PSG recording.
The `SleepEventDataModule` contains the `SleepEventDataset` and is responsible for splitting the data into train, validation and test partitions, and also contain methods to return PyTorch `DataLoader`s for the respective partitions.

The following snippet creates a `SleepEventDataset` in the `SleepEventDataModule` that iterates over 2 train records and yields batches of 16 segments of arousal data.
```python
from nsrr_data.datamodule import SleepEventDataModule

params = dict(
    batch_size=16,
    cache_data=True,
    data_dir="data/processed/mros/ar",
    default_event_window_duration=[3],
    event_buffer_duration=3,
    events={"ar": "Arousal"},
    factor_overlap=2,
    fs=128,
    matching_overlap=0.5,
    minimum_overlap=0.5,
    n_eval=2,
    n_jobs=-1,
    n_records=6,
    n_test=2,
    num_workers=4,
    picks=["c3", "c4", "eogl", "eogr", "chin"],
    scaling="robust",
    seed=1337,
    transform=None,
    window_duration=600,  # seconds
)
datamodule = SleepEventDataModule(**params)
```

## Data preprocessing
Please check out the `README.md` in the `nsrr_data/preprocessing/` directory for instructions on how to convert raw `EDF` to `H5` files.
