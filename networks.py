import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3, resnet101
from torchvision.models.inception import InceptionOutputs


class VanillaCnn(nn.Module):
    def __init__(self,nclasses):
        super(VanillaCnn,self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=4,padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=4,padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=4,padding=1)
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=4,padding=1)
        self.ln1 = nn.Linear(512,128)
        self.ln2 = nn.Linear(128,nclasses)
    
    def forward(self,x):
        #return self.net(x)  # (torch.Size([16, 3, 540, 540]))
        batch_size = x.shape[0]
        #print(x.shape)
        x = F.leaky_relu(self.bn1(self.conv1(x))) #torch.Size([16, 3, 540, 540])
        #print(x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x))) # torch.Size([16, 32, 135, 135])
        #print(x.shape)
        x = F.leaky_relu(self.bn3(self.conv3(x))) # torch.Size([32, 64, 14, 14])
        #print(x.shape)
        x = F.leaky_relu(self.bn4(self.conv4(x))) # 
        #print(x.shape)
        x = F.leaky_relu(self.bn5(self.conv5(x))) # 
        #print(x.shape)
        #x = torch.flatten(x,start_dim=1)
        x = x.view(batch_size,-1)
        #print(x.shape)
        x = F.relu(self.ln1(x)) # torch.Size([16, 3136])
        #print(x.shape)
        x = F.dropout(x,p=0.1)
        return F.relu(self.ln2(x)) # torch.Size([16, 128])

class PreFc(nn.Module):
    def __init__(self,nclasses,premodel):
        super(PreFc,self).__init__()
        self.premodel = premodel
        if premodel == "resnet":
            extract_net = resnet101(weights="DEFAULT")
        elif premodel == "inception":
            extract_net = inception_v3(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError("unsupported pretrained model:{:}".format(premodel))
        extract_net.fc = nn.Identity()
        for para in extract_net.parameters():
            para.require_grads = False
        self.extract_net = extract_net
        d_hid_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(d_hid_dim,512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(512,nclasses),
        )
        self.premodel_name = premodel
    def extract(self,x):
        logits = self.extract_net(x)
        if self.premodel_name == "inception":
            if isinstance(logits,InceptionOutputs):
                logits = logits[0]
            else:
                logits = logits
        else:
            pass
        return logits
    def forward(self,x):
        x_ex = self.extract(x)
        return self.classifier(x_ex)