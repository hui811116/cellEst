# cellEst
Computer Vision and Deep Learning-Based Senescent Cell Identification

## Overview

cellEst is a Python package for preparing microscopy datasets, training image classifiers, evaluating trained models, and generating Grad-CAM visualizations for senescent-cell classification.

The current package workflow is configuration-driven:

- Dataset splits and paths are defined in YAML.
- Models are organized into `cnn` and `transformer` families.
- Training writes model weights and run metadata to a timestamped results directory.
- Inference loads the trained model artifact and its `summary.json` metadata.

The repository also contains older standalone scripts under [examples/scripts](examples/scripts). They remain useful as practical references for dataset preparation and legacy transfer-learning workflows.

## Installation

The project requires Python 3.10 or newer.

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

### Preparing Data

The preprocessing instructions below are unchanged. For implementation details and additional examples, see the scripts in [examples/scripts](examples/scripts), especially [script_pyimageJ_batch_chs.py](examples/scripts/script_pyimageJ_batch_chs.py) and [main_load_doxtreated_dataset.py](examples/scripts/main_load_doxtreated_dataset.py).

1. Convert raw TIFFs to RGB using `pyimageJ_batch_composite.py`:
   ```sh
   python pyimageJ_batch_composite.py cells merged_cells
   ```
2. (Optional) Generate channel–specific composites (nucleus or cytoskeleton):
   ```sh
   python script_pyimageJ_batch_chs.py cells chs_cells [nu/cy] [--recursive]
   ```
   where `nu` selects nucleus and `cy` selects cytoskeleton channels. 
   Include the `--recursive` flag to search for TIFF files recursively in subfolders.

   **Example Folder Structures:**

   *Without `--recursive`* (Images directly in the load folder):
   ```text
   cells/
   ├── r01c01-ch1.tiff
   ├── r01c01-ch2.tiff
   └── ...
   ```

   *With `--recursive`* (Images inside nested subfolders):
   ```text
   cells/
   ├── Day1/
   │   ├── r02c01-ch1.tiff
   │   └── r02c01-ch2.tiff
   └── Day2/
       ├── r03c01-ch1.tiff
       └── r03c01-ch2.tiff
   ```

### Dataset Layout

Training and inference use `torchvision.datasets.ImageFolder`. The configured dataset directory must contain one subdirectory per class:

```text
data/processed/
├── control/
│   └── image_001.tif
└── senescent/
    └── image_002.tif
```

Class folder ordering is stored in `summary.json` and checked during inference.

### Training

The packaged training entry point reads [examples/training/config.yaml](examples/training/config.yaml):

```bash
python examples/training/main.py \
    --config examples/training/config.yaml
```

The configuration contains three sections:

```yaml
data:
  train_data_path: data/processed
  split_ratio: 0.8

model:
  family: cnn
  architecture: convnext_base
  classifier_type: mlp

train:
  batch_size: 16
  epochs: 5
  learning_rate: 0.001
  optimizer: adam
  output_dir: results
```

For a transformer model, use a Hugging Face model identifier:

```yaml
model:
  family: transformer
  architecture: vision_transformer
  pretrained_path: facebook/dino-vitb16
  classifier_type: linear
```

`validation_data_path` and `test_data_path` are optional. When no validation path is supplied, the training dataset is split using `split_ratio`.

Each training run creates a directory containing metrics, plots, `summary.json`, and `model.pt`. The summary stores the validated model configuration, class names, and class count. The model artifact stores the trained state dictionary and metadata required for inference.

### Grad-CAM Inference

Configure [examples/inference/inference.yaml](examples/inference/inference.yaml) with the trained `model.pt`, an `ImageFolder` dataset path, the target class, and an output directory:

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

The scripts in [examples/scripts](examples/scripts) document earlier workflows and are useful for preparing data or reproducing older experiments:

- [script_pyimageJ_batch_chs.py](examples/scripts/script_pyimageJ_batch_chs.py) creates channel-specific composites.

New experiments should use the packaged training and inference entry points above.

## Contact

Dr. Teng-Hui Huang  
<tenghui.huang@sydney.edu.au>
