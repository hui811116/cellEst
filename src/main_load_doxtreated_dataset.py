import os
import sys
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from networks import PreFc
import myutils as uts

parser = argparse.ArgumentParser()
parser.add_argument("folder_path",type=str,help="path/to/image/folder")
parser.add_argument("--save_path",type=str,default="results_split",help="path/to/save/results")
parser.add_argument("--epochs",type=int,default=20,help='number of epochs for training')
parser.add_argument("--batch_size",type=int,default=8,help="minibatch size for training")
parser.add_argument("--lr",type=float,default=1e-4,help="learning rate")
parser.add_argument("--ev_freq",type=int,default=1,
                    help="testing frequency in terms of number of epochs")
parser.add_argument("--seed",type=int,default=42,help="random seed for reproduction")
parser.add_argument("--split",type=float,default=0.9,help="train/test splitting ratio")
parser.add_argument("--model",type=str,default="inception",choices=["resnet","inception"],
                    help="pretrained model to use")
parser.add_argument("--classifier_type",type=str,default="mlp",choices=["mlp","linear"],
                    help="classifier type")

args = parser.parse_args()

trs, tss = uts.get_transforms(args.model)
dataset = ImageFolder(root=args.folder_path, transform=trs)
print(f"labels {np.unique(dataset.targets)}")
print(dataset.classes)

# seed
d_seed = args.seed
print("Setting random seed to:{:}".format(d_seed))
uts.setup_seed(d_seed)

ndata = len(dataset)
train_test_split = args.split
ntrain = int(ndata * train_test_split)
tr_set,ts_set = random_split(dataset,[ntrain , ndata - ntrain])
ts_set.dataset.transform = tss

print(f"Train dataset size:{len(tr_set)}, Test dataset size:{len(ts_set)}")

tr_loader = DataLoader(tr_set,batch_size=args.batch_size,shuffle=True,drop_last=True)
ts_loader = DataLoader(ts_set,batch_size=args.batch_size,shuffle=False)

device = uts.getDevice(False)
print(f"Using model: {args.model} to extract vision features")
network = PreFc(nclasses=len(dataset.classes),premodel=args.model).to(device)
optimizer = torch.optim.Adam(params=network.parameters(),lr=args.lr)

def train(ep):
    """ training loop"""
    network.train()
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
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
    print("Epochs {:}, loss={:.5f}, acc={:.5f}({:}/{:})".format(
        ep,
        loss_sum/len(tr_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt))
    return {"loss":loss_sum/len(tr_loader),"acc":acc_cnt/tot_cnt,"acc_cnt":acc_cnt,"total_cnt":tot_cnt}

def test():
    """ testing loop"""
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    for _, (x_img,y_label) in enumerate(ts_loader):
        x_img = x_img.to(device)
        y_label = y_label.to(device)
        with torch.inference_mode():
            llr_out = network(x_img)
        loss = cr_loss(llr_out,y_label)
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1) == y_label).sum().item()
        tot_cnt += len(y_label)
    print("Testing: loss={:.5f}, Accuracy:{:.5f}({:}/{:})".format(
        loss_sum/len(ts_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt,
    ))
    return {"loss":loss_sum/len(ts_loader),'acc':acc_cnt/tot_cnt,'acc_cnt':acc_cnt,'total_cnt':tot_cnt}

logs_tr = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
logs_ts = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
for e in range(args.epochs):
    tr_log = train(e)
    # TODO: simplify this
    for k,v in tr_log.items():
        if k in logs_tr:
            logs_tr[k].append(v)
    if (e+1)%args.ev_freq == 0 or (e+1) == args.epochs: # last epoch must test
        ts_log = test()
        for kt,vt in ts_log.items():
            if kt in logs_ts:
                logs_ts[kt].append(vt)

tr_logs_df = pd.DataFrame.from_dict(logs_tr)
ts_logs_df = pd.DataFrame.from_dict(logs_ts)
# saving the training logs
save_path_full = os.path.join(os.getcwd(),args.save_path)
os.makedirs(save_path_full,exist_ok=True)
print("Saving logs")
fs_name = f"{args.classifier_type}_split_trts{args.split:.2f}" \
          f"_ep{args.epochs}_bs{args.batch_size}_sd{args.seed}"

with open(os.path.join(save_path_full,fs_name+".pkl"),"wb") as fid:
    pickle.dump({"train":tr_logs_df,'test':ts_logs_df,'args':vars(args)},fid)

# Save the trained model
torch.save(network.state_dict(), os.path.join(save_path_full, fs_name + ".pth"))
print("Model saved to:", os.path.join(save_path_full, fs_name + ".pth"))

print("Done!")
