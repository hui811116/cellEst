import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3, resnet101
from torchvision.models.inception import InceptionOutputs
from ._constants import HIDDEN_DIM
from cellest.classifiers._base_networks import LinearClassifier, MLPClassifier
import logging

logging.basicConfig(level=logging.INFO)

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
        self.premodel = premodel
        for param in self.backbone.parameters():
            param.requires_grad = False
        # replace the fc layer with a new one
        in_features = self.backbone.fc.in_features
        if classifier_type == "mlp":
            self.backbone.fc = MLPClassifier(nclasses,in_features)
        elif classifier_type == "linear":
            self.backbone.fc = LinearClassifier(nclasses,in_features)
        else:
            raise NotImplementedError(f"unsupported classifier type:{classifier_type}")
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    def _pretrained_model_output_format_handle(self,x):
        """handle the output format of the pretrained model, for example, inception_v3 returns a tuple of (logits, aux_logits)"""
        if self.premodel == "inception":
            if isinstance(x,InceptionOutputs):
                x = x[0]
        return x
    def forward(self,x):
        """forward method to extract features from the input image using the pretrained model, 
        and then classify the extracted features using the classifier"""
        # handle model output format
        return self._pretrained_model_output_format_handle(self.backbone(x))