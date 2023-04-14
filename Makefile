.ONESHELL:
.PHONY: create_env setup_ruby

# SHELL=$(SHELL)
CONDA_ACTIVATE=. $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
ENV_NAME=nsrr_data

create_env:
	conda create -n ${ENV_NAME} python=3.10 numpy scikit-learn pytest flake8 black ruby; \
	conda install -y -n ${ENV_NAME} -c conda-forge mne librosa rich ipympl; \
	conda install -y -n ${ENV_NAME} pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia; \
	conda install -y -n ${ENV_NAME} pytorch-lightning -c conda-forge

setup_ruby:
	${CONDA_ACTIVATE} ${ENV_NAME}; \
	conda list; \

download_eegbci_data:
	python -m eegbci.fetch_data \
		   -o $(GROUP_HOME)/data/eegbci \
		   --log
	mkdir -p -v $(SCRATCH)/waveform-conversion/data/raw
	mkdir -p -v $(SCRATCH)/waveform-conversion/data/processed
	ln -s $(GROUP_HOME)/data/eegbci $(SCRATCH)/waveform-conversion/data/raw/eegbci
	ln -s $(SCRATCH)/waveform-conversion/data/raw/eegbci data/raw/eegbci
