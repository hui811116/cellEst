import numpy as np
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from networks import gradCla
import myutils as uts
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", type=str, help="path/to/image/folder")
parser.add_argument("model_path", type=str, help="path/to/trained/model.pth")
parser.add_argument("--save_path", type=str, default="gradCam_processed", help="path/to/save/processed/images")
parser.add_argument("--seed", type=int, default=42, help="random seed for reproduction")
parser.add_argument("--split", type=float, default=0.9, help="train/test splitting ratio")

args = parser.parse_args()

# Setup paths
os.makedirs(args.save_path, exist_ok=True)

# Transforms and Dataset
trs, tss = uts.get_transforms("resnet")
dataset = ImageFolder(root=args.folder_path, transform=trs)

# Seed for deterministic splitting (must match training exactly)
print("Setting random seed to:{:}".format(args.seed))
uts.setup_seed(args.seed)

# Splitting dataset to get the exact same test set
ndata = len(dataset)
train_test_split = args.split
ntrain = int(ndata * train_test_split)
tr_set, ts_set = random_split(dataset, [ntrain, ndata - ntrain])

# Apply test transforms to the test set
ts_set.dataset.transform = tss

print("Test dataset size:{:}".format(len(ts_set)))

device = uts.getDevice(False)

# Load Model
print(f"Loading model from {args.model_path}")
network = gradCla(nclasses=len(dataset.classes)).to(device)
network.load_state_dict(torch.load(args.model_path, map_location=device))
network.eval()

# GradCAM requires gradients to be enabled for the target layers
for param in network.parameters():
    param.requires_grad = True

# Setup GradCAM
target_layers = [network.backbone.layer4[-1]]
cam = GradCAM(model=network, target_layers=target_layers)

# Identify the target class index for 'senescent'
if 'senescent' in dataset.class_to_idx:
    senescent_idx = dataset.class_to_idx['senescent']
else:
    print("Warning: 'senescent' class not found in dataset. Defaulting to index 1.")
    senescent_idx = 1
    
targets = [ClassifierOutputTarget(senescent_idx)]
print(f"Generating GradCAM for '{dataset.classes[senescent_idx]}' class (index {senescent_idx}).")

# Normalization constants used during training/transforms
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Process all images in the test set
for rnd_idx in range(len(ts_set)):
    input_tensor, true_label = ts_set[rnd_idx]
    
    # Get original file path to use its name for saving
    original_path = ts_set.dataset.samples[ts_set.indices[rnd_idx]][0]
    filename = os.path.basename(original_path)
    
    # Add batch dimension
    input_tensor_batch = input_tensor.unsqueeze(0).to(device)
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor_batch, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert tensor back to image format (HWC) for visualization
    rgb_img = input_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Reverse the normalization
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Predict the label
    with torch.no_grad():
        output = network(input_tensor_batch)
        pred_label = output.argmax(dim=1).item()
        
    true_class_name = dataset.classes[true_label]
    pred_class_name = dataset.classes[pred_label]

    # Save the resulting image
    save_file = os.path.join(args.save_path, f"gradcam_{filename}.tiff")
    
    # Create figure with title
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(visualization)
    ax.set_title(f"True: {true_class_name} | Pred: {pred_class_name}")
    ax.axis('off')
    
    # Save figure
    fig.savefig(save_file, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    if (rnd_idx + 1) % 10 == 0:
        print(f"Processed {rnd_idx + 1}/{len(ts_set)} images...")

print(f"Done! All images saved to '{os.path.abspath(args.save_path)}'")
