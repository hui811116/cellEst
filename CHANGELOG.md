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