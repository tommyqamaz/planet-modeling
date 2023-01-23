DVC_REMOTE_NAME := modeling
USERNAME := soldatovman

.PHONY: test
test:
	@poetry run pytest

.PHONY: lint
lint:
	@poetry run flake8 .

.PHONY: install
install:
	pip install poetry && poetry --version && poetry config virtualenvs.in-project true && poetry install -vv

.PHONY: train
train:
	@poetry run python -m src.train

.PHONY: test-coverage
test-coverage:
	@poetry run pytest --cov-report html --cov-report term-missing --cov=src/ tests/

.PHONY: install_c_libs
install_c_libs:
	apt-get update && apt-get install -y --no-install-recommends gcc ffmpeg libsm6 libxext6

.PHONY: download_weights
download_weights:
	@poetry run dvc pull weights pvtv2.onnx


.PHONY: init_dvc
init_dvc:
	dvc init --no-scm
	dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(USERNAME)/dvc_files
	dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
	dvc config cache.type hardlink,symlink
