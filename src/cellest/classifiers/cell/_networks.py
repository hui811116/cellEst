"""CNN classifiers with a shared preprocessing and GradCAM interface."""

import numpy as np
import torch.nn as nn
from torchvision.models import (
    Inception_V3_Weights,
    ResNet101_Weights,
    ConvNeXt_Base_Weights,
    EfficientNet_V2_M_Weights,
    inception_v3,
    resnet101,
    convnext_base,
    efficientnet_v2_m,
)
from torchvision.models.inception import InceptionOutputs

from cellest.classifiers._base_networks import LinearClassifier, MLPClassifier


class CNNClassifier(nn.Module):
    """A frozen ImageNet CNN backbone with a trainable classification head."""

    def __init__(self, nclasses: int, architecture: str, classifier_type: str = "mlp"):
        super().__init__()
        self.family = "cnn"
        self.architecture = architecture
        self.backbone = self._build_backbone(architecture)
        self._replace_classifier(classifier_type, nclasses)
        self._freeze_backbone()

    @staticmethod
    def _build_backbone(architecture: str):
        if architecture == "resnet101":
            return resnet101(weights=ResNet101_Weights.DEFAULT)
        if architecture == "inception_v3":
            return inception_v3(weights=Inception_V3_Weights.DEFAULT)
        if architecture == "convnext_base":
            return convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        if architecture == "efficientnet_v2_m":
            return efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        raise ValueError(f"Unsupported CNN architecture: {architecture}")

    @staticmethod
    def _build_head(classifier_type: str, feature_dim: int, nclasses: int):
        if classifier_type == "linear":
            return LinearClassifier(nclasses, feature_dim)
        if classifier_type == "mlp":
            return MLPClassifier(nclasses, feature_dim)
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def _replace_classifier(self, classifier_type: str, nclasses: int):
        if self.architecture in {"resnet101", "inception_v3"}:
            feature_dim = self.backbone.fc.in_features
            head = self._build_head(classifier_type, feature_dim, nclasses)
            setattr(self.backbone, "fc", head)
            self._classifier = head
            return

        if self.architecture == "convnext_base":
            feature_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = self._build_head(
                classifier_type, feature_dim, nclasses
            )
            self._classifier = self.backbone.classifier[-1]
            return

        if self.architecture == "efficientnet_v2_m":
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = self._build_head(
                classifier_type, feature_dim, nclasses
            )
            self._classifier = self.backbone.classifier[1]
            return

        raise ValueError(f"Unsupported CNN architecture: {self.architecture}")

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self._classifier.parameters():
            parameter.requires_grad = True

    @property
    def target_layers(self):
        return self.get_target_layers()

    def get_target_layers(self):
        if self.architecture == "resnet101":
            return [self.backbone.layer4[-1]]
        if self.architecture == "convnext_base":
            return [self.backbone.features[-1][-1]]
        if self.architecture == "efficientnet_v2_m":
            return [self.backbone.features[-1]]
        if self.architecture == "inception_v3":
            return [self.backbone.Mixed_7c]
        raise ValueError(f"Unsupported CNN architecture: {self.architecture}")

    @property
    def reshape_transform(self):
        return None

    def preprocess(self, images, device):
        return images.to(device)

    def display_images(self, inputs):
        mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225])[None, :, None, None]
        images = inputs.detach().cpu().numpy()
        return np.clip((images * std + mean).transpose(0, 2, 3, 1), 0, 1)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        if isinstance(outputs, InceptionOutputs):
            return outputs.logits
        return outputs


class PreFc(CNNClassifier):
    """Compatibility wrapper for the legacy ``premodel`` argument."""

    def __init__(self, nclasses: int, premodel: str = "resnet", classifier_type: str = "mlp"):
        architecture = {
            "resnet": "resnet101",
            "inception": "inception_v3",
            "convnext": "convnext_base",
            "efficientnet": "efficientnet_v2_m",
        }.get(premodel, premodel)
        super().__init__(nclasses, architecture, classifier_type)


GradCla = PreFc
