from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn


class HFTransformerGradCAM(nn.Module):
	"""Adapt a Hugging Face vision transformer classifier for Grad-CAM."""

	def __init__(self, classifier):
		super().__init__()
		self.backbone = classifier.model
		self.classifier = classifier.classifier
		self.processor = classifier.processor
		self.patch_size = self._pair(self.backbone.config.patch_size)
		self.num_prefix_tokens = 1 + getattr(
			self.backbone.config, "num_register_tokens", 0
		)
		self._grid_size = None
		self.target_layer = self._find_last_encoder_layer()

	@staticmethod
	def _pair(value):
		if isinstance(value, Sequence):
			return tuple(value)
		return value, value

	def _find_last_encoder_layer(self):
		encoder = getattr(self.backbone, "encoder", None)
		for layer_name in ("layer", "layers"):
			layers = getattr(encoder, layer_name, None)
			if layers is not None and len(layers):
				last_layer = layers[-1]
				for norm_name in ("norm1", "layernorm_before"):
					target_layer = getattr(last_layer, norm_name, None)
					if isinstance(target_layer, nn.Module):
						return target_layer
				raise ValueError(
					"The final encoder block has no supported tensor-output norm layer. "
					"Select a tensor-output target layer for this architecture."
				)
		raise ValueError(
			"Could not find the Hugging Face transformer encoder blocks. "
			"Select a target layer explicitly for this model architecture."
		)

	def preprocess(self, images):
		"""Convert PIL images or raw image tensors to processor pixel values."""
		pixel_values = self.processor(images, return_tensors="pt").pixel_values
		device = next(self.classifier.parameters()).device
		return pixel_values.to(device)

	def forward(self, pixel_values):
		height, width = pixel_values.shape[-2:]
		patch_height, patch_width = self.patch_size
		self._grid_size = (
			height // patch_height,
			width // patch_width,
		)
		outputs = self.backbone(pixel_values)
		cls_embedding = outputs.last_hidden_state[:, 0, :]
		return self.classifier(cls_embedding)

	def reshape_transform(self, token_activations):
		"""Drop prefix tokens and reshape patch tokens to BCHW feature maps."""
		if self._grid_size is None:
			raise RuntimeError("Run a forward pass before reshaping transformer activations.")
		if not isinstance(token_activations, torch.Tensor):
			raise TypeError("The target layer must return a tensor of token activations.")

		batch_size, token_count, channels = token_activations.shape
		grid_height, grid_width = self._grid_size
		patch_tokens = token_activations[:, self.num_prefix_tokens :, :]
		expected_tokens = grid_height * grid_width
		if patch_tokens.shape[1] != expected_tokens:
			raise ValueError(
				f"Expected {expected_tokens} patch tokens, got {patch_tokens.shape[1]}. "
				"Check the model's prefix-token count and patch size."
			)

		patch_maps = patch_tokens.reshape(
			batch_size, grid_height, grid_width, channels
		)
		return patch_maps.permute(0, 3, 1, 2)

	# def generate(self, images, target_classes: int | Sequence[int]):
	# 	"""Generate one Grad-CAM map per image for the requested class index."""
	# 	try:
	# 		from pytorch_grad_cam import GradCAM
	# 		from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
	# 	except ImportError as error:
	# 		raise ImportError(
	# 			"Grad-CAM support requires the 'grad-cam' package. "
	# 			"Install project requirements before generating maps."
	# 		) from error

	# 	pixel_values = self.preprocess(images).requires_grad_(True)
	# 	if isinstance(target_classes, int):
	# 		class_indices = [target_classes] * pixel_values.shape[0]
	# 	else:
	# 		class_indices = list(target_classes)
	# 	if len(class_indices) != pixel_values.shape[0]:
	# 		raise ValueError("Pass one target class index per image in the batch.")

	# 	targets = [ClassifierOutputTarget(index) for index in class_indices]
	# 	self.eval()
	# 	with GradCAM(
	# 		model=self,
	# 		target_layers=[self.target_layer],
	# 		reshape_transform=self.reshape_transform,
	# 	) as cam:
	# 		return cam(input_tensor=pixel_values, targets=cast(Any, targets))
