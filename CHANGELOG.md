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
