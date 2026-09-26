import transformers
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification, TorchAoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerCla(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base", num_classes: int = 2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        model = transformers.AutoModel.from_pretrained(model_name, device_map="auto")
        for param in model.parameters():
            param.requires_grad = False  # Freeze the pre-trained model parameters
        self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.processor(x, return_tensors="pt").pixel_values
        x = x.to(self.model.device)  # Move input to the same device as the model
        outputs = self.model(x)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits