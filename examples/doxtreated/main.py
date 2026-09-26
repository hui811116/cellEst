import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

from cellest.config import CellestClaTrainConfig
from cellest.classifiers import (
    get_device,
    setup_seed,
    get_transforms,
    binary_curves,
    metrics_all,
    train_classifier,
    evaluate_classifier,
    save_run_outputs,
    create_output_dir,
    save_run_plots,
)
from cellest.classifiers.cell import PreFc
from cellest.classifiers.pretrained import TransformerCla
from cellest.classifiers import build_optimizer, build_scheduler
from cellest.dataloader import build_datasets

from cellest.logging import TrainLogger

logger = TrainLogger(__name__)

def build_model(model_cfg, nclasses):
    """instantiate the classifier described by the validated model config"""
    if model_cfg.arch == "cell":
        return PreFc(nclasses, model_cfg.premodel, model_cfg.classifier_type)
    return TransformerCla(model_cfg.pretrained_path, nclasses)


def evaluate_model(model, loader, classes, criterion, device):
    loss, accuracy, logits, labels = evaluate_classifier(
        model, loader, criterion, device, return_predictions=True
    )
    metrics = metrics_all(logits, labels)
    probabilities = torch.softmax(logits, dim=1)
    if len(classes) != 2:
        logger.warning(
            "Binary curves requested for %d classes; using class 0 as positive.", len(classes)
        )
        labels = (labels == 0).long()
        probabilities = probabilities[:, 0]
    else:
        probabilities = probabilities[:, 1]
    curves = binary_curves(labels.numpy(), probabilities.numpy())
    return loss, accuracy, metrics, curves


def build_data_loaders(config, device):
    transform_train, transform_test = get_transforms(config.model.arch)
    train_set, validation_set, test_set = build_datasets(
        config.data, transform_train, transform_test
    )
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_set, batch_size=config.train.batch_size, shuffle=False
    )
    test_loader = (
        DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False)
        if test_set is not None
        else None
    )
    classes = train_set.dataset.classes
    logger.log_dataset_summary(
        train_set,
        validation_set,
        test_set,
        train_set.dataset.class_to_idx,
        device,
    )
    return train_set, validation_set, test_set, train_loader, validation_loader, test_loader, classes


def train_epochs(model, train_loader, validation_loader, optimizer, scheduler, config, criterion, device):
    history = []
    for epoch in range(config.train.epochs):
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_logits, val_labels = evaluate_classifier(
            model, validation_loader, criterion, device, return_predictions=True
        )
        val_metrics = metrics_all(val_logits, val_labels)
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_auc": val_metrics["roc"],
                "val_prc": val_metrics["prc"],
                "val_f1": val_metrics["f1"],
                "val_recall": val_metrics["recall"],
            }
        )

        logger.log_epoch_summary(
            epoch + 1,
            config.train.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_metrics["roc"],
        )
    return history


def run_training(config, args):
    setup_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)
    (
        train_set,
        validation_set,
        test_set,
        train_loader,
        validation_loader,
        test_loader,
        classes,
    ) = build_data_loaders(config, device)
    model = build_model(config.model, len(classes)).to(device)
    optimizer = build_optimizer(config.train, model)
    scheduler = build_scheduler(config.train, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    source_name, run_name, output_dir = create_output_dir(config)

    history = train_epochs(
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        config,
        criterion,
        device,
    )
    evaluation_loader = test_loader or validation_loader
    evaluation_label = "test" if test_loader is not None else "validation"
    evaluation_loss, evaluation_acc, evaluation_metrics, curves = evaluate_model(
        model, evaluation_loader, classes, criterion, device
    )

    model_size = config.model.pretrained_path or config.model.premodel or config.model.arch
    model_metadata = {
        "source_name": source_name,
        "head_type": config.model.classifier_type,
        "model_size": model_size,
        "arch": config.model.arch,
        "pretrained_path": config.model.pretrained_path,
        "premodel": config.model.premodel,
        "num_classes": len(classes),
        "class_names": classes,
    }

    summary = {
        "run_name": run_name,
        "dataset": config.data.train_data_path.name,
        "model_size": model_size,
        "channels": 3,
        "head": config.model.classifier_type,
        "seed": args.seed,
        "evaluation_split": evaluation_label,
        "evaluation_auc": evaluation_metrics["roc"] if math.isfinite(evaluation_metrics["roc"]) else None,
        "evaluation_acc": evaluation_acc,
        "evaluation_loss": evaluation_loss,
        "evaluation_prc": evaluation_metrics["prc"],
        "evaluation_f1": evaluation_metrics["f1"],
        "evaluation_recall": evaluation_metrics["recall"],
        "n_train": len(train_set),
        "n_val": len(validation_set),
        "n_test": len(test_set) if test_set is not None else 0,
        "class_names": classes,
    }
    save_run_outputs(output_dir, history, curves, model_metadata, summary)
    save_run_plots(history, curves, classes, run_name, evaluation_label, output_dir)
    logger.log_run_summary(output_dir, summary)
    return summary


def main(args):
    """Load configuration, train/evaluate, and save run outputs."""
    logger.configure_logging(args)
    config = CellestClaTrainConfig.from_yaml(Path(args.config))
    return run_training(config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example script for training cellEst Models")
    parser.add_argument("--config", type=str, required=True, help="path/to/config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--force_cpu", action="store_true", help="force training on CPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("--log", type=str, default=None, help="path/to/logfile.log")
    args = parser.parse_args()
    main(args)