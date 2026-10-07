import abc
import torch.nn as nn
from torch.nn import functional as F
from ._constants import HIDDEN_DIM


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