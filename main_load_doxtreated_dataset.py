import numpy as np
import pandas as pd
import os
import sys
import pickle
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from networks import VanillaCnn,PreFc
import torch.nn as nn
import myutils as uts
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("folder_path",type=str,help="path/to/image/folder")
parser.add_argument("--save_path",type=str,default="results_split",help="path/to/save/results")
parser.add_argument("--epochs",type=int,default=20,help='number of epochs for training')
parser.add_argument("--batch_size",type=int,default=8,help="minibatch size for training")
parser.add_argument("--lr",type=float,default=1e-4,help="learning rate")
parser.add_argument("--ev_freq",type=int,default=1,help="testing frequency in terms of number of epochs")
parser.add_argument("--seed",type=int,default=42,help="random seed for reproduction")
parser.add_argument("--split",type=float,default=0.9,help="train/test splitting ratio")

#argv = sys.argv
args = parser.parse_args()

#folder_path = args.folder_path

"""
# inception
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
trs = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tss = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
trs = transforms.Compose([
    transforms.Resize((540, 540)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])

tss = transforms.Compose([
    transforms.Resize((540, 540)),
    transforms.ToTensor(),
])
"""
dataset = ImageFolder(root=args.folder_path, transform=trs)
#print(len(dataset))
print("labels {:}".format(np.unique(dataset.targets)))
print(dataset.classes)
#loader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

# seed
use_seed =True
if use_seed:
    d_seed = args.seed
    print("Setting random seed to:{:}".format(d_seed))
    #torch.manual_seed(d_seed)
    uts.setup_seed(d_seed)
else:
    print("Random data splitting, may not be reproduced")

ndata = len(dataset)
train_test_split = args.split
ntrain = int(ndata * train_test_split)
tr_set,ts_set = random_split(dataset,[ntrain , ndata - ntrain])
ts_set.dataset.transform = tss

print("Train dataset size:{:}, Test dataset size:{:}".format(len(tr_set),len(ts_set)))

#batch_size = 8
#epochs = 20
#ev_freq= 1
#learning_rate = 1e-4
tr_loader = DataLoader(tr_set,batch_size=args.batch_size,shuffle=True,drop_last=True)
ts_loader = DataLoader(ts_set,batch_size=args.batch_size,shuffle=False)

device = uts.getDevice(False)
#network = VanillaCnn(nclasses=len(dataset.classes)).to(device)
network = PreFc(nclasses=len(dataset.classes),premodel='inception').to(device)
optimizer = torch.optim.Adam(params=network.parameters(),lr=args.lr)

def train(ep):
    network.train()
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    for bi, (X,Y) in enumerate(tr_loader):
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        llr_out = network(X)
        loss = cr_loss(llr_out,Y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1)== Y).sum().item()
        tot_cnt += len(Y)
    print("Epochs {:}, loss={:.5f}, acc={:.5f}({:}/{:})".format(
        ep,
        loss_sum/len(tr_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt))
    #return None
    return {"loss":loss_sum/len(tr_loader),"acc":acc_cnt/tot_cnt,"acc_cnt":acc_cnt,"total_cnt":tot_cnt}

def test():
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    for bi, (X,Y) in enumerate(ts_loader):
        X = X.to(device)
        Y = Y.to(device)
        with torch.inference_mode():
            llr_out = network(X)
        loss = cr_loss(llr_out,Y)
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1) == Y).sum().item()
        tot_cnt += len(Y)
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
        if k in logs_tr.keys():
            logs_tr[k].append(v)
    if (e+1)%args.ev_freq == 0 or (e+1) == args.epochs: # last epoch must test
        ts_log = test()
        for kt,vt in ts_log.items():
            if kt in logs_ts.keys():
                logs_ts[kt].append(vt)

tr_logs_df = pd.DataFrame.from_dict(logs_tr)
ts_logs_df = pd.DataFrame.from_dict(logs_ts)
# saving the training logs
save_path_full = os.path.join(os.getcwd(),args.save_path)
os.makedirs(save_path_full,exist_ok=True)
print("Saving logs")
fs_name = "split_trts{:.2f}_ep{:}_bs{:}_sd{:}".format(args.split,args.epochs,args.batch_size,args.seed)

with open(os.path.join(save_path_full,fs_name+".pkl"),"wb") as fid:
    pickle.dump({"train":tr_logs_df,'test':ts_logs_df,'args':vars(args)},fid)

# TODO: add model saving afterward
print("Done!")
