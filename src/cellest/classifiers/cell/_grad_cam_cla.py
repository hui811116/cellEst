"""Backward-compatible name for the unified CNN classifier."""

from ._networks import CNNClassifier


class GradCla(CNNClassifier):
    """Compatibility wrapper for the legacy ``premodel`` argument."""

    def __init__(self, nclasses, premodel="resnet", classifier_type="mlp"):
        architecture = {
            "resnet": "resnet101",
            "inception": "inception_v3",
            "convnext": "convnext_base",
            "efficientnet": "efficientnet_v2_m",
        }.get(premodel, premodel)
        super().__init__(nclasses, architecture, classifier_type)
