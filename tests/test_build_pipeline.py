import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from cellest.classifiers import (
    binary_curves,
    build_optimizer,
    build_scheduler,
    evaluate_classifier,
    metrics_all,
    save_run_outputs,
    train_classifier,
)
from cellest.config import CellestClaTrainConfig
from cellest.dataloader import build_datasets

rng = np.random.default_rng(1)


class TinyNet(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, nclasses))

    def forward(self, x):
        return self.classifier(x)


@pytest.fixture
def config(tmp_path):
    lines = ["url,class"]
    for label in ("a", "b"):
        for i in range(8):
            name = f"{label}{i}.png"
            Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(tmp_path / name)
            lines.append(f"{name},{label}")
    (tmp_path / "train.csv").write_text("\n".join(lines))
    (tmp_path / "config.yaml").write_text(
        "data:\n  dataset_base_path: .\n  train_data_file: train.csv\n"
        "model:\n  family: cnn\n  architecture: resnet101\n"
        "train:\n  batch_size: 4\n  epochs: 1\n  learning_rate: 0.01\n"
        "  optimizer: adam\n  output_dir: out\n"
    )
    return CellestClaTrainConfig.from_yaml(tmp_path / "config.yaml")


def test_build_train_evaluate_and_save(config, tmp_path):
    to_tensor = transforms.ToTensor()
    train_set, val_set, test_set = build_datasets(config.data, to_tensor, to_tensor)
    assert test_set is None
    classes = train_set.dataset.classes
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    model = TinyNet(len(classes))
    optimizer = build_optimizer(config.train, model)
    build_scheduler(config.train, optimizer)
    criterion = nn.CrossEntropyLoss()
    train_classifier(model, train_loader, optimizer, criterion, torch.device("cpu"))

    loss, acc, logits, labels = evaluate_classifier(
        model, val_loader, criterion, torch.device("cpu"), return_predictions=True
    )
    metrics = metrics_all(logits, labels)
    probs = torch.softmax(logits, dim=1)[:, 1]
    curves = binary_curves(labels.numpy(), probs.numpy())

    out = tmp_path / "out"
    out.mkdir()
    summary = {
        "dataset_base_path": config.data.dataset_base_path,  # a Path must be serializable
        "evaluation_acc": acc,
        "evaluation_f1": metrics["f1"],
        "class_names": classes,
    }
    save_run_outputs(
        out,
        {"epoch": [1], "train_loss": [0.0]},
        curves,
        {"state_dict": model.state_dict(), "class_names": classes},
        summary,
    )
    saved = json.loads((out / "summary.json").read_text())
    assert saved["dataset_base_path"] == str(tmp_path.resolve())
    for name in ("history.csv", "roc.csv", "pr.csv", "confmat.npy", "model.pt"):
        assert (out / name).exists()
