# cellEst
Computer Vision and Deep Learning–Based Senescent Cell Identification

## Overview

This repository provides utilities for converting microscopy data into a format
suitable for PyTorch and training deep models to distinguish senescent cells.

### Key Concepts

- **Data format** – Raw 16‑bit TIFF images are converted to RGB `.tif` using
  PyImageJ utilities.
- **Dataset handling** – We use `torchvision.datasets.ImageFolder` to build
  datasets where each subfolder corresponds to a class label.
- **Feature extraction** – Pre‑trained CNNs (ResNet or Inception) are used as
  backbone extractors.
- **Classifier** – Extracted features are fed to a small fully connected network
  that produces logits for the target classes.
- **Scripts provided**:
  1. `main_load_doxtreated_dataset.py` – Splits a single image folder into
     training and testing subsets, trains a model, and saves logs/arguments.
  2. `main_transfer_test.py` – Performs transfer evaluation using separate
     training and testing folders.

## Installation

> _Assumes you have Python 3.8+ and PyTorch installed.  See `requirements.txt`
> if provided._

```bash
pip install -r requirements.txt  # if available
```

## Usage

### Preparing Data

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

### Training Scripts

- **Split training/testing from one folder**
  ```sh
  python main_load_doxtreated_dataset.py /path/to/images [--options]
  ```
  This script will save a pickle file containing logs and the parsed
  arguments (including the chosen pretrained model).

- **Transfer evaluation with explicit folders**
  ```sh
  python main_transfer_test.py /path/to/train /path/to/test [--options]
  ```
  *Ensure that class subfolders exist in both train and test directories and
  that their names match.*

### Selecting the Pretrained Model

Both scripts accept a `--model` flag with two choices:
```
--model {resnet,inception}
```
- `resnet` uses ResNet101 with 224×224 input size.
- `inception` (default) uses Inception‑V3 with 299×299 input size.

You can explicitly set the model at training time:
```sh
python main_load_doxtreated_dataset.py /data/images --model resnet
```

When running `main_transfer_test.py`, you can either specify the model again or
load it from a previous training pickle:
```sh
python main_transfer_test.py train_dir test_dir --load_args_from
    results_split/split_trts0.90_ep20_bs8_sd42.pkl
```
In the latter case the model choice stored in the saved arguments is applied
automatically.

To override a loaded model, simply include `--model` on the command line again.

### Examining Options

Run any script without arguments to view all available flags and defaults:
```sh
python main_load_doxtreated_dataset.py
python main_transfer_test.py
```

## Contact

Dr. Teng‑Hui Huang  
<tenghui.huang@sydney.edu.au>
