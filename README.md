# cellEst
Computer Vision and Deep Learning-Based Senescent Cell Identification

## Overview

cellEst is a Python package for preparing microscopy datasets, training image classifiers, evaluating trained models, and generating Grad-CAM visualizations for senescent-cell classification.

The current package workflow is configuration-driven:

- Datasets are defined in YAML, either as image folders or as a `.csv`/`.json` table of image urls and classes.
- Models are organized into `cnn` and `transformer` families.
- Training writes model weights and run metadata to a timestamped results directory.
- Inference loads the trained model artifact and its `summary.json` metadata.

The repository also contains a preprocessing script and an older transfer-learning script under [examples/scripts](examples/scripts).

## Installation

The project requires Python 3.12 or newer.

```bash
pip install -e .
```

This installs the package with portable pip-compatible core dependencies. To install the full pinned environment, which uses CUDA when available and falls back to CPU otherwise:

```bash
pip install -r requirements.txt
```

## Supported Architectures

### CNN

The CNN family uses torchvision ImageNet backbones with a trainable `linear` or `mlp` classification head:

- `resnet101` with 224x224 inputs
- `inception_v3` with 299x299 inputs
- `convnext_base` with 224x224 inputs
- `efficientnet_v2_m` with 480x480 inputs

### Transformer

The transformer family loads Hugging Face vision backbones through `transformers`. The supported architecture identifier is `vision_transformer`; the concrete model is selected by `pretrained_path`.

Examples include:

- `facebook/dino-vitb16`
- `google/vit-base-patch16-224`

Both families expose the same classifier and Grad-CAM integration used by the training and inference examples.

## Usage

The workflow is: (1) preprocess raw TIFFs, (2) organize the dataset, (3) optionally describe it as a `.csv` file, (4) write a `config.yaml`, (5) train, (6) run Grad-CAM inference. Run every command from the repository root.

### 1. Preprocessing Raw Images (optional)

[examples/scripts/script_pyimageJ_batch_chs.py](examples/scripts/script_pyimageJ_batch_chs.py) pairs the channel TIFFs (`-ch1` nucleus, `-ch2` cytoskeleton) and merges them with ImageJ:

```sh
python examples/scripts/script_pyimageJ_batch_chs.py <load_path> <save_path> <nu|cy|composite> [--recursive]
```

- `nu`: nucleus channel, `cy`: cytoskeleton channel, `composite`: both merged into one RGB image.
- `--recursive` searches subfolders; the folder structure is kept under `<save_path>`.

```sh
python examples/scripts/script_pyimageJ_batch_chs.py cells merged_cells composite --recursive
```

Input layouts (without and with `--recursive`):

```text
cells/                      cells/
├── r01c01-ch1.tiff         ├── Day1/
├── r01c01-ch2.tiff         │   ├── r02c01-ch1.tiff
└── ...                     │   └── r02c01-ch2.tiff
                            └── Day2/ ...
```

### 2. Organizing the Dataset

Two dataset formats are supported. Choose one per training run.

**A. Image folders** (`train_data_path`): one subdirectory per class, read with `torchvision.datasets.ImageFolder`.

```text
data/processed/
├── control/
│   └── image_001.tif
└── senescent/
    └── image_002.tif
```

**B. Table file** (`train_data_file`): a `.csv` or `.json` file listing every image and its class. Images can live anywhere, and one class can combine several folders. Use this when classes are spread over several directories or the images are remote.

- The **first column** is the image `url`/`uri` (the header must be named `url` or `uri`, case-insensitive).
- The **second column** is the class name.
- A uri is a local path (absolute, or relative to `dataset_base_path`) or an `http`/`https`/`ftp`/`file` URL.

```csv
uri,class_name
processed/adscs_take_3_processed/P12/r02c05f59p01_type_composite.tiff,natural_senescent
processed/p4_dox_2026_merge/senescent/r03c01f12p01_type_composite.tiff,chemical_senescent
https://example.org/images/control_001.png,control
```

The equivalent `.json` is a list of objects whose first key is the url/uri and second key the class:

```json
[
  {"uri": "processed/p4_dox_2026_merge/control/img_001.tiff", "class_name": "control"},
  {"uri": "https://example.org/images/control_002.png", "class_name": "control"}
]
```

Class indices are the sorted class names of the training file. Validation and test files must only use classes found in the training file.

### 3. Creating the `.csv` from a Data Card

[examples/dataset/main_csv_dataset.py](examples/dataset/main_csv_dataset.py) scans folders listed in a YAML data card and writes the two-column CSV (paths are written as given, relative to the directory you run it from):

```yaml
# examples/dataset/datacard.yaml
name: experiment_natural_versus_chemical_senescent
description: A dataset comparing natural and chemical senescent processes.
classes:
  - label:
      name: natural_senescent
      paths:
        - processed/adscs_take_3_processed/P12
  - label:
      name: control
      paths:
        - processed/adscs_take_3_processed/P4
        - processed/p4_dox_2026_merge/control
```

```sh
python examples/dataset/main_csv_dataset.py examples/dataset/datacard.yaml --output_folder examples/dataset
```

This writes `examples/dataset/<name>.csv` (images with `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` extensions, found recursively) and prints the class distribution. To use separate train/test sets, build one data card (and CSV) for each.

### 4. Writing `config.yaml`

[examples/training/config.yaml](examples/training/config.yaml) has three sections. Example using a table file:

```yaml
data:
  dataset_base_path: "/abs_path/to/dataset/folder"          # use absolute path to avoid confusion
  train_data_file: "examples/dataset/experiment_natural_versus_chemical_senescent.csv"
  #validation_data_file: "examples/dataset/validation.csv"  # optional
  #test_data_file: "examples/dataset/test.csv"              # optional
  split_ratio: 0.8                    # used when no validation file/path is given

model:
  family: transformer
  architecture: vision_transformer
  pretrained_path: facebook/dino-vitb16   # or google/vit-base-patch16-224
  classifier_type: linear

train:
  batch_size: 16
  epochs: 5
  learning_rate: 0.001
  optimizer: adam
  output_dir: results
```

Rules for the `data` section:

- Give exactly one of `train_data_path` (image folders) or `train_data_file` (`.csv`/`.json`). The same format must be used for all splits; mixing is rejected.
- `dataset_base_path` applies to table files only. A relative value is resolved against the folder containing `config.yaml` (not the working directory or the script location). It is joined to relative `*_data_file` paths and to every relative `uri` in the files. Absolute paths and URLs are used unchanged.
- Files are validated at load time: supported extension (`.csv`, `.json`), existence, the `url`/`uri` + class header, and non-empty rows.
- Without a validation path/file, the training data is split with `split_ratio` (stratified by class).

Using image folders instead:

```yaml
data:
  train_data_path: data/processed
  #validation_data_path: data/validation   # optional
  #test_data_path: data/test               # optional
  split_ratio: 0.8
```

For a CNN model:

```yaml
model:
  family: cnn
  architecture: convnext_base   # resnet101 | inception_v3 | convnext_base | efficientnet_v2_m
  classifier_type: mlp          # mlp | linear
```

### 5. Training

```bash
python examples/training/main.py --config examples/training/config.yaml
```

Options: `--seed 42`, `--force_cpu`, `-v/--verbose`, `--log path/to/logfile.log`.

Each run creates a directory under `output_dir` containing `history.csv`, `roc.csv`, `pr.csv`, `confmat.npy`, plots, `summary.json` and `model.pt`. The summary stores the validated model configuration, class names and class count. `model.pt` stores the trained state dictionary and the metadata required for inference.

### 6. Testing

```bash
pip install pytest
python -m pytest tests -q
```

[tests/test_config.py](tests/test_config.py) covers both dataset config variants and dataset building; [tests/test_build_pipeline.py](tests/test_build_pipeline.py) runs a small offline config-to-saved-outputs pipeline.

### 7. Grad-CAM Inference

Configure [examples/inference/inference.yaml](examples/inference/inference.yaml) with the trained `model.pt`, an `ImageFolder` dataset path (inference currently reads image folders only), the target class, and an output directory:

```yaml
model_path: results/cnn_convnext_base_run/model.pt
data:
  dataset_path: data/processed
output_dir: gradcam_output
target_class: senescent
batch_size: 4
threshold: 0.5
```

Run inference from the repository root so relative paths are resolved from the current working directory:

```bash
python examples/inference/main.py \
    --config examples/inference/inference.yaml
```

Use `--force_cpu` when GPU inference is unavailable:

```bash
python examples/inference/main.py \
    --config examples/inference/inference.yaml \
    --force_cpu
```

For every input image, inference writes a side-by-side visualization:

- Left: the model-sized image with the Grad-CAM contour.
- Right: the same model-sized image with the Grad-CAM heatmap overlay.

The output preserves the dataset class subdirectory structure.

### Legacy Example Scripts

[examples/scripts/main_load_doxtreated_dataset.py](examples/scripts/main_load_doxtreated_dataset.py) is an earlier standalone transfer-learning script kept for reproducing older experiments. New experiments should use the training and inference entry points above.

## Contact

Dr. Teng-Hui Huang  
<tenghui.huang@sydney.edu.au>
