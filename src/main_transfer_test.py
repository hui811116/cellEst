import numpy as np
import pandas as pd
import os
import sys
import pickle
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from networks import VanillaCnn,PreFc
import torch.nn as nn
import myutils as uts
import argparse
import evaluate as ev

parser = argparse.ArgumentParser()
parser.add_argument("train_datapath",type=str,help="path/to/training/image/folder")
parser.add_argument("test_datapath",type=str,help="path/to/testing/image/folder")
parser.add_argument("--epochs",type=int,default=10,help="training epochs")
parser.add_argument("--batch_size",type=int,default=16,help="mini batch size for training")
parser.add_argument("--eval_freq",type=int,default=1,help="evaluation frequency, epochs per testing")
parser.add_argument("--lr",type=float,default=1e-4,help="learning rate of training")
parser.add_argument("--seed",type=int,default=42,help="random seed for reproduction")
parser.add_argument("--save_path",type=str,default="results_transfer",help="path/to/save/results")
parser.add_argument("--model",type=str,default="inception",choices=["resnet","inception"],help="pretrained model to use")
parser.add_argument("--load_args_from",type=str,default=None,help="path/to/pkl/file to load model choice from previous training")
args = parser.parse_args()

# Load model choice from previous training arguments if provided
if args.load_args_from is not None:
    print(f"Loading arguments from {args.load_args_from}")
    try:
        with open(args.load_args_from, "rb") as fid:
            saved_data = pickle.load(fid)
            if "args" in saved_data:
                saved_args = saved_data["args"]
                if "model" in saved_args:
                    args.model = saved_args["model"]
                    print(f"Loaded model choice: {args.model}")
            else:
                print("Warning: 'args' key not found in saved file")
    except Exception as e:
        print(f"Error loading arguments from {args.load_args_from}: {e}")

trs, tss = uts.get_transforms(args.model)
tr_set = ImageFolder(root=args.train_datapath, transform=trs)
#print(len(tr_set))
print("labels {:}".format(np.unique(tr_set.targets)))
print(tr_set.classes)
# 
#tr_class_map = tr_set.class_to_idx
print("Training map:")
print(tr_set.class_to_idx)

ts_set = ImageFolder(root=args.test_datapath,transform=tss)
#loader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)

print("testing map:")
print(ts_set.class_to_idx)
assert ts_set.class_to_idx == tr_set.class_to_idx
# NOTE:make sure the mapping of labels is the same
# seed
# use_seed =False
# if use_seed:
#     d_seed = 123 # TODO: change the seed for reproduction
#     print("Setting random seed to:{:}".format(d_seed))
#     torch.manual_seed(d_seed)
# else:
#     print("Random data splitting, may not be reproduced")
uts.setup_seed(args.seed) # NOTE: for reproduction

#ndata = len(dataset)
#train_test_split = 0.9
#ntrain = int(ndata * train_test_split)
#tr_set,ts_set = random_split(dataset,[ntrain , ndata - ntrain])
#ts_set.dataset.transform = tss
#print("Train dataset size:{:}, Test dataset size:{:}".format(len(tr_set),len(ts_set)))

#batch_size = args.batch_size
#epochs = args.epochs
#ev_freq= args.eval_freq
#learning_rate = args.lr
tr_loader = DataLoader(tr_set,batch_size=args.batch_size,shuffle=True,drop_last=True)
ts_loader = DataLoader(ts_set,batch_size=args.batch_size,shuffle=False)

device = uts.getDevice(False)
#network = VanillaCnn(nclasses=len(dataset.classes)).to(device)
network = PreFc(nclasses=len(ts_set.classes),premodel=args.model).to(device)
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
    return {"loss":loss_sum/len(tr_loader),"acc":acc_cnt/tot_cnt,'acc_cnt':acc_cnt,"total_cnt":tot_cnt}

def test():
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    y_pred = []
    y_true = []
    network.eval()
    for bi, (X,Y) in enumerate(ts_loader):
        X = X.to(device)
        Y = Y.to(device)
        with torch.inference_mode():
            llr_out = network(X)
        loss = cr_loss(llr_out,Y)
        loss_sum += loss.item()
        y_pred.append(F.softmax(llr_out,dim=1).detach().cpu())
        y_true.append(Y.detach().cpu())
        acc_cnt += (llr_out.argmax(dim=1) == Y).sum().item()
        tot_cnt += len(Y)
    y_pred_all = torch.cat(y_pred,dim=0)
    y_true_all = torch.cat(y_true,dim=0)
    ev_dict = ev.metrics_all(y_pred_all,y_true_all)
    print("Testing: loss={:.5f}, Accuracy:{:.5f}({:}/{:})".format(
        loss_sum/len(ts_loader),
        acc_cnt/tot_cnt,
        acc_cnt,
        tot_cnt,
    ))
    return {"loss":loss_sum/len(ts_loader),'acc':acc_cnt/tot_cnt,'acc_cnt':acc_cnt,"total_cnt":tot_cnt,'metrics':ev_dict}

# loggings
log_tr = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
log_ts = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}
ts_metrics = {}
for e in range(args.epochs):
    tr_dict = train(e)
    for k,v in tr_dict.items():
        if k in log_tr.keys():
            log_tr[k].append(v)
    if (e+1)% args.eval_freq == 0 or (e+1) == args.epochs: # test for the last epoch
        ts_dict =test()
        for k,v in ts_dict.items():
            if k in log_ts.keys():
                log_ts[k].append(v)
            elif k == "metrics":
                for ek,evv in v.items():
                    if ek not in ts_metrics.keys():
                        ts_metrics[ek] = []
                    ts_metrics[ek].append(evv)


# Saving the results
tr_df = pd.DataFrame.from_dict(log_tr)
ts_df = pd.DataFrame.from_dict(log_ts)
ev_df = pd.DataFrame.from_dict(ts_metrics)
full_path = os.path.join(os.getcwd(),args.save_path)
os.makedirs(full_path,exist_ok=True)
fs_name = "transfer_ep{:}_bs{:}_sd{:}".format(args.epochs,args.batch_size,args.seed)
with open(os.path.join(full_path,fs_name+".pkl"),'wb') as fid:
    pickle.dump({"train":log_tr,"test":log_ts,'metrics':ev_df,'args':vars(args)},fid)

print("Done!")