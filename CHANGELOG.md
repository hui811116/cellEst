# Changelog

## Unreleased

### Added
- Integrated DVC for MLOps workflows, enabling data and model version tracking and reproducible pipelines.

### Changed
- Disabled GitHub Actions to reduce cloud resource usage and save infrastructure costs.
- Change the folder structure. Now all .py files are in ```src/```.

## [0.0.1] - 2026-09-21
- Integrate ```src/script_pyimage_batch_chs.py``` script, now support nucleus/cytoskeleton/both
- Creating src module folders ```src/preprocess``` and ```src/classifiers``` for imageJ preprocessing and pytorch ML classifier training


## [0.1.1] - 2026-09-26
- Packaging cellest as a classification package supporting HuggingFace pretrained model
- Upgrade source file structure for packaging
- Configuration change: now one configure the data paths, model names, and training parameters using a config.yaml


## [0.2.1] - 2026-09-27
- New design of the architectures, now supports CNN and Transformers based backbone
- Added example scripts using this package. ```examples/training/main.py``` and ```examples/inference/main.py```
- GradCam function tested with the example scripts


## [0.2.2] - 2026-10-28
### Added
- New way to create datasets from a table file (`.csv` or `.json`) instead of an image-folder tree: set `train_data_file` and, optionally, `validation_data_file` / `test_data_file` in the `data` config.
- Table format: the first column is the image `url`/`uri` and the second column is the class name. Local paths and `http`/`https`/`ftp`/`file` URLs are supported.
- `dataset_base_path` option: relative `*_data_file` paths and relative `uri`s are joined to it. When loaded from YAML, a relative value is relative to the folder containing `config.yaml`.
- `ImageTableView` / `ImageTableSource` (`cellest.dataloader`), mirroring `ImageFolderView`, plus `ImageTableView.from_file()`. Class indices of the validation/test files follow the training file; unknown classes raise an error.
- Tests for both config variants, dataset building and an end-to-end build/train/save pipeline (`tests/test_config.py`, `tests/test_build_pipeline.py`).

### Changed
- `DataConfig` validation: exactly one of `train_data_path` or `train_data_file` is required, the two modes cannot be mixed across splits, and data files are checked for a supported extension and the `url`/`uri` + class header.
- `examples/training/main.py` supports both dataset modes.

### Fixed
- `summary.json` saving no longer fails on `Path` and NumPy values.
- Transformer target-layer lookup is more tolerant of the layer layout across `transformers` versions.