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
from ._constants import HIDDEN_DIM
from cellest.classifiers._base_networks import LinearClassifier, MLPClassifier
import logging

logging.basicConfig(level=logging.INFO)

def _get_extract_net(premodel):
    """get the pretrained model for feature extraction"""
    if premodel == "resnet":
        extract_net = resnet101(weights="DEFAULT")
    elif premodel == "inception":
        extract_net = inception_v3(weights="IMAGENET1K_V1")
    else:
        raise NotImplementedError(f"unsupported pretrained model:{premodel}")
    extract_net.fc = nn.Identity()
    for para in extract_net.parameters():
        para.requires_grad = False
    return extract_net


class PreFc(nn.Module):
    """a classifier with pretrained model for feature extraction, 
    """
    def __init__(self,nclasses,premodel,classifier_type="mlp"):
        super(PreFc,self).__init__()
        self.premodel = premodel
        extract_net = _get_extract_net(premodel)
        if premodel == "resnet":
            extract_net = resnet101(weights="DEFAULT")
        elif premodel == "inception":
            extract_net = inception_v3(weights="IMAGENET1K_V1")
        else:
            raise NotImplementedError(f"unsupported pretrained model:{premodel}")
        extract_net.fc = nn.Identity()
        for para in extract_net.parameters():
            para.requires_grad = False
        self.extract_net = extract_net
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


