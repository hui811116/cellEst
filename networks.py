"""
This module defines the networks used for classification,
including a simple linear classifier
"""
import abc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3, resnet101
from torchvision.models.inception import InceptionOutputs


HIDDEN_DIM = 2048

def get_extract_net(premodel):
    """get the pretrained model for feature extraction"""
    if premodel == "resnet":
        extract_net = resnet101(weights="DEFAULT")
    elif premodel == "inception":
        extract_net = inception_v3(weights="IMAGENET1K_V1")
    else:
        raise NotImplementedError(f"unsupported pretrained model:{premodel}")
    extract_net.fc = nn.Identity()
    for para in extract_net.parameters():
        para.require_grads = False
    return extract_net

class ClassifierBase(nn.Module):
    """base class for classifier, the input dimension is determined by 
    the pretrained model used for feature extraction, and 
    the output dimension is the number of classes"""
    def __init__(self,nclasses):
        super(ClassifierBase,self).__init__()
        self.nclasses = nclasses
    @abc.abstractmethod
    def forward(self,x):
        """forward method to be implemented by subclasses"""
        raise NotImplementedError("forward method not implemented")

class LinearClassifier(ClassifierBase):
    """a simple linear classifier, the input dimension is determined by the 
    pretrained model used for feature extraction, and 
    the output dimension is the number of classes"""
    def __init__(self, nclasses, input_dim=HIDDEN_DIM):
        super(LinearClassifier, self).__init__(nclasses)
        self.classifier = nn.Linear(input_dim, nclasses)
    def forward(self, x):
        return self.classifier(x)

class MLPClassifier(ClassifierBase):
    """a simple MLP classifier with one hidden layer, the input dimension is determined by 
    the pretrained model used for feature extraction, and 
    the output dimension is the number of classes"""
    def __init__(self, nclasses, input_dim=HIDDEN_DIM):
        super(MLPClassifier, self).__init__(nclasses)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(512, nclasses),
        )
    def forward(self, x):
        return self.classifier(x)

class PreFc(nn.Module):
    """a classifier with pretrained model for feature extraction, 
    """
    def __init__(self,nclasses,premodel,classifier_type="mlp"):
        super(PreFc,self).__init__()
        self.premodel = premodel
        if premodel == "resnet":
            extract_net = resnet101(weights="DEFAULT")
        elif premodel == "inception":
            extract_net = inception_v3(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError(f"unsupported pretrained model:{premodel}")
        extract_net.fc = nn.Identity()
        for para in extract_net.parameters():
            para.require_grads = False
        self.extract_net = extract_net
        #d_hid_dim = 2048
        #self.classifier = nn.Sequential(
        #    nn.Linear(d_hid_dim,512),
        #    nn.Dropout(p=0.1),
        #    nn.ReLU(),
        #    nn.Linear(512,nclasses),
        #)
        if classifier_type == "mlp":
            self.classifier = MLPClassifier(nclasses,HIDDEN_DIM)
        elif classifier_type == "linear":
            self.classifier = LinearClassifier(nclasses,HIDDEN_DIM)
        else:
            raise NotImplementedError(f"unsupported classifier type:{classifier_type}")
        self.premodel_name = premodel
    def extract(self,x):
        """extract features from the input image using the pretrained model, and 
        return the extracted features"""
        logits = self.extract_net(x)
        if self.premodel_name == "inception":
            if isinstance(logits,InceptionOutputs):
                logits = logits[0]
        else:
            pass
        return logits
    def forward(self,x):
        """forward method to extract features from the input image using the pretrained model, 
        and then classify the extracted features using the classifier"""
        x_ex = self.extract(x)
        return self.classifier(x_ex)


class GradCla(nn.Module):
    """a classifier with pretrained model for feature extraction, 
    the input dimension is determined by the pretrained model used for feature extraction, and 
    the output dimension is the number of classes"""
    def __init__(self,nclasses,premodel="resnet",classifier_type="mlp"):
        super(GradCla,self).__init__()
        if premodel == "resnet":
            self.backbone = resnet101(weights="DEFAULT")
        elif premodel == "inception":
            self.backbone = inception_v3(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError(f"unsupported pretrained model:{premodel}")
        #self.backbone = resnet101(weights="DEFAULT")
        for param in self.backbone.parameters():
            param.require_grads = False
        # replace the fc layer with a new one
        in_features = self.backbone.fc.in_features
        if classifier_type == "mlp":
            self.backbone.fc = MLPClassifier(nclasses,in_features)
        elif classifier_type == "linear":
            self.backbone.fc = LinearClassifier(nclasses,in_features)
        else:
            raise NotImplementedError(f"unsupported classifier type:{classifier_type}")
        #self.backbone.fc = nn.Sequential(
        #    nn.Linear(in_features,512),
        #    nn.Dropout(p=0.1),
        #    nn.ReLU(),
        #    nn.Linear(512,nclasses),
        #)
        for param in self.backbone.fc.parameters():
            param.require_grads = True
    def forward(self,x):
        """forward method to extract features from the input image using the pretrained model, 
        and then classify the extracted features using the classifier"""
        return self.backbone(x)
