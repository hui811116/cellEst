from types import SimpleNamespace

import torch
from torch import nn

from cellest.classifiers._metrics import metrics_all
from cellest.classifiers._training import save_training_checkpoint, train_classifier
from cellest.classifiers.pretrained._grad_cam_hf import HFTransformerGradCAM
from cellest.classifiers.pretrained._transformer_cla import TransformerClassifier


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()


class DummyTransformerBackbone(nn.Module):
    def __init__(self, model_type="vit", include_target_layer=True):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            patch_size=2,
            model_type=model_type,
            num_register_tokens=0,
        )
        self.encoder = SimpleNamespace(layer=nn.ModuleList([DummyBlock()]))
        if not include_target_layer:
            del self.encoder.layer[-1].norm1

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_tokens = 4
        tokens = torch.arange(
            batch_size * (patch_tokens + 1) * self.config.hidden_size,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        ).reshape(batch_size, patch_tokens + 1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)


class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyTransformerBackbone()
        self.classifier = nn.Linear(8, 2)
        self.processor = lambda images, return_tensors: SimpleNamespace(
            pixel_values=torch.ones(len(images), 3, 4, 4)
        )


class DummyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        self._classifier = nn.Linear(4, 2)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        return self._classifier(self.backbone(inputs))


class DummyFinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.classifier(self.backbone(inputs))


def test_metrics_all_supports_multiclass_scores():
    logits = torch.tensor(
        [[6.0, 1.0, 0.5], [0.5, 5.0, 0.5], [0.5, 0.25, 5.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    metrics = metrics_all(logits, labels)

    assert metrics["acc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0
    assert 0.0 <= metrics["prc"] <= 1.0
    assert 0.0 <= metrics["roc"] <= 1.0


def test_metrics_all_returns_nan_for_single_class_binary_validation():
    logits = torch.tensor([[4.0, 1.0], [5.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.long)

    metrics = metrics_all(logits, labels)

    assert torch.isnan(torch.tensor(metrics["prc"]))
    assert torch.isnan(torch.tensor(metrics["roc"]))


def test_train_classifier_keeps_frozen_backbone_in_eval_mode(tmp_path):
    model = DummyTrainModel()
    optimizer = torch.optim.SGD(model._classifier.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = [(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))]

    train_classifier(model, loader, optimizer, criterion, torch.device("cpu"))

    assert model.backbone.training is False
    assert model._classifier.training is True

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(checkpoint_path, model, optimizer, None, 1, 1, 0.9)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["classifier_state_dict"].keys() == model._classifier.state_dict().keys()


def test_train_classifier_keeps_trainable_backbone_in_train_mode():
    model = DummyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = [(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))]

    train_classifier(model, loader, optimizer, criterion, torch.device("cpu"))

    assert model.backbone.training is True
    assert model.classifier.training is True


def test_hf_transformer_gradcam_uses_classifier_backbone():
    classifier = DummyClassifier()

    adapter = HFTransformerGradCAM(classifier)
    output = adapter(torch.ones(2, 3, 4, 4))
    feature_maps = adapter.reshape_transform(torch.ones(2, 5, 8))

    assert adapter.backbone is classifier.backbone
    assert output.shape == (2, 2)
    assert feature_maps.shape == (2, 8, 2, 2)


def test_transformer_classifier_uses_requested_head_and_rejects_swin(monkeypatch):
    processor = SimpleNamespace(
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )

    monkeypatch.setattr(
        "cellest.classifiers.pretrained._transformer_cla.AutoImageProcessor.from_pretrained",
        lambda _: processor,
    )
    monkeypatch.setattr(
        "cellest.classifiers.pretrained._transformer_cla.transformers.AutoModel.from_pretrained",
        lambda _: DummyTransformerBackbone(),
    )

    classifier = TransformerClassifier("facebook/dino-vitb16", 3, "MLP")

    assert classifier.classifier.__class__.__name__ == "MLPClassifier"

    monkeypatch.setattr(
        "cellest.classifiers.pretrained._transformer_cla.transformers.AutoModel.from_pretrained",
        lambda _: DummyTransformerBackbone(model_type="swin"),
    )

    try:
        TransformerClassifier("microsoft/swin-tiny-patch4-window7-224", 3, "linear")
    except ValueError as error:
        assert "ViT-compatible model" in str(error)
    else:
        raise AssertionError("Expected Swin backbones to be rejected.")

    monkeypatch.setattr(
        "cellest.classifiers.pretrained._transformer_cla.transformers.AutoModel.from_pretrained",
        lambda _: DummyTransformerBackbone(include_target_layer=False),
    )

    try:
        TransformerClassifier("google/vit-base-patch16-224", 3, "linear")
    except ValueError as error:
        assert "tensor-output transformer target layer" in str(error)
    else:
        raise AssertionError("Expected unsupported encoder layouts to be rejected.")
