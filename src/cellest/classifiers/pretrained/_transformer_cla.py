"""Hugging Face vision-transformer classifier with GradCAM support."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import transformers
from transformers import AutoImageProcessor

from cellest.classifiers._base_networks import LinearClassifier, MLPClassifier


SUPPORTED_MODEL_TYPES = {"dinov2", "vit"}


class TransformerClassifier(nn.Module):
    """A frozen Hugging Face vision backbone with a trainable classifier head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        classifier_type: str = "linear",
    ):
        super().__init__()
        self.family = "transformer"
        self.architecture = "vision_transformer"
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = transformers.AutoModel.from_pretrained(model_name)
        self._validate_backbone()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.classifier = self._build_head(
            classifier_type, self.backbone.config.hidden_size, num_classes
        )
        self.patch_size = self._pair(self.backbone.config.patch_size)
        self.num_prefix_tokens = 1 + getattr(
            self.backbone.config, "num_register_tokens", 0
        )
        self._grid_size = None

    @staticmethod
    def _pair(value):
        if isinstance(value, Sequence):
            return tuple(value)
        return value, value

    @staticmethod
    def _build_head(classifier_type: str, feature_dim: int, num_classes: int):
        if classifier_type == "linear":
            return LinearClassifier(num_classes, feature_dim)
        if classifier_type == "mlp":
            return MLPClassifier(num_classes, feature_dim)
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def _validate_backbone(self):
        model_type = getattr(self.backbone.config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                "Unsupported transformer backbone. Use a ViT-compatible model such as "
                "'facebook/dino-vitb16' or 'google/vit-base-patch16-224'."
            )
        if not hasattr(self.backbone.config, "hidden_size") or not hasattr(
            self.backbone.config, "patch_size"
        ):
            raise ValueError(
                "Unsupported transformer backbone configuration. The model must expose "
                "'hidden_size' and 'patch_size' for classifier and Grad-CAM support."
            )

    @property
    def target_layers(self):
        return self.get_target_layers()

    def get_target_layers(self):
        encoder = getattr(self.backbone, "encoder", None)
        for name in ("layer", "layers"):
            layers = getattr(encoder, name, None)
            if layers:
                block = layers[-1]
                for norm_name in ("norm1", "layernorm_before"):
                    layer = getattr(block, norm_name, None)
                    if isinstance(layer, nn.Module):
                        return [layer]
        raise ValueError("Could not find a tensor-output transformer target layer.")

    @property
    def reshape_transform(self):
        return self._reshape_transform

    def preprocess(self, images, device=None):
        pixel_values = self.processor(images, return_tensors="pt").pixel_values
        return pixel_values if device is None else pixel_values.to(device)

    def display_images(self, inputs):
        mean = torch.tensor(self.processor.image_mean, device=inputs.device)[None, :, None, None]
        std = torch.tensor(self.processor.image_std, device=inputs.device)[None, :, None, None]
        images = (inputs.detach() * std + mean).clamp(0, 1)
        return images.cpu().numpy().transpose(0, 2, 3, 1)

    def forward(self, pixel_values):
        height, width = pixel_values.shape[-2:]
        patch_height, patch_width = self.patch_size
        self._grid_size = (height // patch_height, width // patch_width)
        outputs = self.backbone(pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)

    def _reshape_transform(self, token_activations):
        if self._grid_size is None:
            raise RuntimeError("Run a forward pass before reshaping activations.")
        if not isinstance(token_activations, torch.Tensor):
            raise TypeError("The target layer must return tensor activations.")

        batch_size, _, channels = token_activations.shape
        grid_height, grid_width = self._grid_size
        patch_tokens = token_activations[:, self.num_prefix_tokens :, :]
        expected_tokens = grid_height * grid_width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}."
            )
        patch_maps = patch_tokens.reshape(
            batch_size, grid_height, grid_width, channels
        )
        return patch_maps.permute(0, 3, 1, 2)


# Compatibility alias for existing imports.
TransformerCla = TransformerClassifier
