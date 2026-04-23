"""
Training script for models that support GradCAM visualization
"""
import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import myutils as uts
from networks import GradCla

parser = argparse.ArgumentParser()
parser.add_argument("folder_path",type=str,help="path/to/image/folder")
parser.add_argument("--save_path",type=str,default="results_split",help="path/to/save/results")
parser.add_argument("--epochs",type=int,default=5,help='number of epochs for training')
parser.add_argument("--batch_size",type=int,default=8,help="minibatch size for training")
parser.add_argument("--lr",type=float,default=1e-4,help="learning rate")
parser.add_argument("--ev_freq",type=int,default=1,
                    help="testing frequency in terms of number of epochs")
parser.add_argument("--seed",type=int,default=42,help="random seed for reproduction")
parser.add_argument("--split",type=float,default=0.9,help="train/test splitting ratio")
parser.add_argument('--classifier_type',type=str,choices=['mlp','linear'],default="mlp",
                    help="Choose classifier type")
args = parser.parse_args()

trs, tss = uts.get_transforms("resnet")
dataset = ImageFolder(root=args.folder_path, transform=trs)
print(f"labels {np.unique(dataset.targets)}")
print(dataset.classes)

# seed
d_seed = args.seed
print(f"Setting random seed to:{d_seed}")
uts.setup_seed(d_seed)

ndata = len(dataset)
train_test_split = args.split
ntrain = int(ndata * train_test_split)
tr_set,ts_set = random_split(dataset,[ntrain , ndata - ntrain])
ts_set.dataset.transform = tss

print("Train dataset size:{:}, Test dataset size:{:}".format(len(tr_set),len(ts_set)))

tr_loader = DataLoader(tr_set,batch_size=args.batch_size,shuffle=True,drop_last=True)
ts_loader = DataLoader(ts_set,batch_size=args.batch_size,shuffle=False)


device = uts.getDevice(False)
# We should change to support multiple premodel [fixme]
print("Using model: resnet101 to extract vision features") 

network = GradCla(nclasses=len(dataset.classes),
                  premodel="resnet",
                  classifier_type=args.classifier_type).to(device)
optimizer = torch.optim.Adam(params=network.parameters(),lr=args.lr)

def train(ep):
    """Training loop"""
    network.train()
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    t_start = time.perf_counter()
    for _, (x_img,y_label) in enumerate(tr_loader):
        optimizer.zero_grad()
        x_img = x_img.to(device)
        y_label = y_label.to(device)
        llr_out = network(x_img)
        loss = cr_loss(llr_out,y_label)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1)== y_label).sum().item()
        tot_cnt += len(y_label)
    t_end = time.perf_counter()
    print("Epochs {:}, loss={:.5f}, acc={:.5f}({:}/{:}), time={:.2f}s".format(
        ep,
        loss_sum/len(tr_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt,
        t_end - t_start))
    return {"loss":loss_sum/len(tr_loader),"acc":acc_cnt/tot_cnt,"acc_cnt":acc_cnt,"total_cnt":tot_cnt}

def test():
    """Testing loop"""
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    t_start = time.perf_counter()
    for _, (x_img,y_label) in enumerate(ts_loader):
        x_img = x_img.to(device)
        y_label = y_label.to(device)
        with torch.inference_mode():
            llr_out = network(x_img)
        loss = cr_loss(llr_out,y_label)
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1) == y_label).sum().item()
        tot_cnt += len(y_label)
    t_end = time.perf_counter()
    print("Testing: loss={:.5f}, Accuracy:{:.5f}({:}/{:}), time={:.2f}s".format(
        loss_sum/len(ts_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt,
        t_end - t_start
    ))
    return {"loss":loss_sum/len(ts_loader),'acc':acc_cnt/tot_cnt,'acc_cnt':acc_cnt,'total_cnt':tot_cnt}

logs_tr = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
logs_ts = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
for e in range(args.epochs):
    tr_log = train(e)
    # FIXME: simplify this block as stacks of for loops are not easy to follow
    for k,v in tr_log.items():
        if k in logs_tr:
            logs_tr[k].append(v)
    if (e+1)%args.ev_freq == 0 or (e+1) == args.epochs: # last epoch must test
        ts_log = test()
        for kt,vt in ts_log.items():
            if kt in logs_ts:
                logs_ts[kt].append(vt)

network.eval()
for param in network.parameters():
    param.require_grads = True

# loading images to process with gradcam
rnd_idx = 0 # replace with specific index if needed
input_tensor, _ = ts_set[rnd_idx] # get the first image from the test set
test_image_path = ts_set.dataset.samples[ts_set.indices[rnd_idx]][0] # get the path of the test image

input_tensor = input_tensor.unsqueeze(0).to(device) # add batch dimension and move to device

target_layers = [network.backbone.layer4[-1]]
cam = GradCAM(model=network, target_layers=target_layers)

targets=[ClassifierOutputTarget(1)] # senescent cells
# generate heapmap
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
# visualize the heatmap
rgb_img = input_tensor.cpu().numpy().transpose(0,2,3,1)[0] # convert to HWC format, first image index is 0 since we only have one image in the batch
# reverse the normalization (assuming ImageNet normalization)
# use the resnet mean and std for denormalization, this should be 
# modified if using different pretrained models with different normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
rgb_img = std * rgb_img + mean
rgb_img = np.clip(rgb_img, 0, 1) # clip to [0,1] range
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# display the visualization

# saving the miscillaneous results
tr_logs_df = pd.DataFrame.from_dict(logs_tr)
ts_logs_df = pd.DataFrame.from_dict(logs_ts)
# saving the training logs
save_path_full = os.path.join(os.getcwd(),args.save_path)
os.makedirs(save_path_full,exist_ok=True)
print("Saving logs")
fs_name = f"gradcla_{args.classifier_type}" \
          f"_trts{args.split:.2f}_ep{args.epochs}_bs{args.batch_size}_sd{args.seed}"

with open(os.path.join(save_path_full,fs_name+".pkl"),"wb") as fid:
    pickle.dump({"train":tr_logs_df,'test':ts_logs_df,'args':vars(args)},fid)

# Save the trained model
torch.save(network.state_dict(), os.path.join(save_path_full, fs_name + ".pth"))
print("Model saved to:", os.path.join(save_path_full, fs_name + ".pth"))
print("Done!")
