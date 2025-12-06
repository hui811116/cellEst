# cellEst
Computer Vision and Deep Learning Based Senescent Cell Identification

## Overview

- Data format: use PyImageJ to convert 16-bytes .tiff to RGB .tif images 

- Dataset: we use pytorch built-in dataset utility ImageFolder to create dataset objects

- Algorithm: we leverage pre-trained models for visual feature extraction

- Classifier: the extracted features becomes the input to neural networks with output as the logits

- Training/Testing: We provide to scripts:
  1) main_load_doxtreated_dataset.py: spliting the same image folder into training and testing
  2) main_transfer_test.py: two input arguments indicating the training and test datasets

## Usage

- Generate .tif RGB images
  Assume a directory of raw images (16bytes .tiff) named "cells"
  Suppose that the path to save the RGB images is named "merged_cells"
  ```
  python pyimageJ_batch_composite.py cells merged_cells
  ```
  In addition, if you want to generate channel specific RGB images, and want to store it to "chs_cells"
  We provide two options: 1) nuclius and 2) cytoskeleton
  ```
  python script_pyimageJ_batch_chs.py cells chs_cells [nu/cy]
  ```
  the last argument will indicate which type of channel images to store. "nu" for nuclius and "cy" for cytoskeleton.
- Run the scripts for training
  1) Train/Test Split using the same dataset
  ```
  python main_load_doxtreated_dataset.py /path/to/image/folder [--options]
  ```
  3) Specify the directories for training and testing sets
  ```
  python main_transfer_test.py /path/to/training/images /path/to/testing/images [--options]
  ```
  You can see the options by runing the script without arguments.
  * Please make sure that the folders (which will be treated as labels) in the /path/to/training/images and /path/to/testing/images are the same.

## Contact
  Dr. Teng-Hui Huang
  tenghui.huang@sydney.edu.au
