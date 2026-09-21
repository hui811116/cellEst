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
parser.add_argument('--premodel',type=str,choices=['resnet','inception'],default="resnet",
                    help="Choose pre-trained model for feature extraction")


def parse_args():
    """Parse command-line arguments."""
    return parser.parse_args()


def to_log_dataframe(logs):
    """Convert training/test logs to a DataFrame for both scalar and list values."""
    if not logs:
        return pd.DataFrame()

    if all(np.isscalar(v) for v in logs.values()):
        return pd.DataFrame([logs])

    return pd.DataFrame(logs)

def train(model,dataloader,optimizer,device,epochs):
    """Training loop"""
    tr_logs = {"loss":[],"acc":[],"acc_cnt":[],"total_cnt":[]}

    def train_step(ep):
        """Training loop"""
        model.train()
        loss_sum = 0
        acc_cnt = 0
        tot_cnt = 0
        cr_loss = nn.CrossEntropyLoss(reduction='sum')
        t_start = time.perf_counter()
        for _, (x_img,y_label) in enumerate(dataloader):
            optimizer.zero_grad()
            x_img = x_img.to(device)
            y_label = y_label.to(device)
            llr_out = model(x_img)
            loss = cr_loss(llr_out,y_label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            acc_cnt += (llr_out.argmax(dim=1)== y_label).sum().item()
            tot_cnt += len(y_label)
        t_end = time.perf_counter()

        print(f"Epochs {ep}/{epochs}, loss={loss_sum/len(dataloader):.5f},",
              f" acc={acc_cnt/tot_cnt:.5f}({acc_cnt}/{tot_cnt}),",
              f" time={t_end - t_start:.2f}s")
        return {"loss":loss_sum/len(dataloader),"acc":acc_cnt/tot_cnt,
                "acc_cnt":acc_cnt,"total_cnt":tot_cnt}
    for e in range(epochs):
        #tr_logs.append(train_step(e))
        logs = train_step(e)
        for k,v in logs.items():
            if k in tr_logs:
                tr_logs[k].append(v)
    return tr_logs


def test(model,dataloader,device):
    """Testing loop"""
    model.eval()
    loss_sum = 0
    acc_cnt = 0
    tot_cnt = 0
    cr_loss = nn.CrossEntropyLoss(reduction='sum')
    t_start = time.perf_counter()
    for _, (x_img,y_label) in enumerate(dataloader):
        x_img = x_img.to(device)
        y_label = y_label.to(device)
        with torch.inference_mode():
            llr_out = model(x_img)
        loss = cr_loss(llr_out,y_label)
        loss_sum += loss.item()
        acc_cnt += (llr_out.argmax(dim=1) == y_label).sum().item()
        tot_cnt += len(y_label)
    t_end = time.perf_counter()
    print(f"Testing: loss={loss_sum/len(dataloader):.5f},",
          f" Accuracy:{acc_cnt/tot_cnt:.5f}({acc_cnt}/{tot_cnt}),",
          f" time={t_end-t_start:.2f}s")
    return {"loss":loss_sum/len(dataloader),'acc':acc_cnt/tot_cnt,
            'acc_cnt':acc_cnt,'total_cnt':tot_cnt}

def main(arg):
    """Main function"""
    trs, tss = uts.get_transforms(arg.premodel)
    dataset = ImageFolder(root=arg.folder_path, transform=trs)
    print(f"labels {np.unique(dataset.targets)}")
    print(dataset.classes)
    print("Label mapping:")
    print(dataset.class_to_idx)
    # seed
    d_seed = arg.seed
    print(f"Setting random seed to:{d_seed}")
    uts.setup_seed(d_seed)

    ndata = len(dataset)
    train_test_split = arg.split
    ntrain = int(ndata * train_test_split)
    tr_set,ts_set = random_split(dataset,[ntrain , ndata - ntrain])
    ts_set.dataset.transform = tss

    print(f"Train dataset size:{len(tr_set)}, Test dataset size:{len(ts_set)}")

    tr_loader = DataLoader(tr_set,batch_size=arg.batch_size,shuffle=True,drop_last=True)
    ts_loader = DataLoader(ts_set,batch_size=arg.batch_size,shuffle=False)


    device = uts.getDevice(False)
    # We should change to support multiple premodel [fixme]
    print(f"Using model: {arg.premodel} to extract vision features") 

    network = GradCla(nclasses=len(dataset.classes),
                    premodel=arg.premodel,
                    classifier_type=arg.classifier_type).to(device)
    optimizer = torch.optim.Adam(params=network.parameters(),lr=arg.lr)

    # Classifier training and testing
    logs_tr = train(network,tr_loader,optimizer,device, arg.epochs)
    logs_ts = test(network,ts_loader,device)

    # grad cam preparation
    # saving the miscillaneous results
    tr_logs_df = to_log_dataframe(logs_tr)
    ts_logs_df = to_log_dataframe(logs_ts)
    # saving the training logs
    save_path_full = os.path.join(os.getcwd(),arg.save_path)
    os.makedirs(save_path_full,exist_ok=True)
    print("Saving logs")
    fs_name = f"gradcla_{arg.classifier_type}" \
            f"_pre_{arg.premodel}" \
            f"_trts{arg.split:.2f}_ep{arg.epochs}_bs{arg.batch_size}_sd{arg.seed}"

    with open(os.path.join(save_path_full,fs_name+".pkl"),"wb") as fid:
        pickle.dump({"train":tr_logs_df,'test':ts_logs_df,'args':vars(arg),'label_map':dataset.class_to_idx},fid)

    # Save the trained model
    torch.save(network.state_dict(), os.path.join(save_path_full, fs_name + ".pth"))
    print("Model saved to:", os.path.join(save_path_full, fs_name + ".pth"))
    print("Done!")



if __name__ == "__main__":
    #GradCam supported classifier training script
    # take parsed arguments including training information and paths
    # output a trained model and a pickle file with the training arguments for reproduction
    main(parse_args())
