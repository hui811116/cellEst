import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cellest.config import CellestClaTrainConfig
from cellest.classifiers import (
    get_device,
    setup_seed,
    get_transforms,
    train_classifier,
    evaluate_classifier,
)
from cellest.classifiers.cell import PreFc
from cellest.classifiers.pretrained import TransformerCla
from cellest.classifiers import build_optimizer, build_scheduler
from cellest.dataloader import build_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(model_cfg, nclasses):
    """instantiate the classifier described by the validated model config"""
    if model_cfg.arch == "cell":
        return PreFc(nclasses, model_cfg.premodel, model_cfg.classifier_type)
    return TransformerCla(model_cfg.pretrained_path, nclasses)


def main(args):
    """train a cellEst classifier from a validated YAML config"""
    config = CellestClaTrainConfig.from_yaml(Path(args.config))

    setup_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)

    transform_train, transform_test = get_transforms(config.model.arch)

    train_set, test_set = build_datasets(config.data, transform_train, transform_test)
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False)

    classes = train_set.dataset.classes if hasattr(train_set, "dataset") else train_set.classes
    model = build_model(config.model, len(classes)).to(device)

    optimizer = build_optimizer(config.train, model)
    scheduler = build_scheduler(config.train, optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.train.epochs):
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_classifier(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        logger.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            epoch + 1, config.train.epochs, train_loss, train_acc, test_loss, test_acc,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example script for training cellEst Models")
    parser.add_argument("--config", type=str, required=True, help="path/to/config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--force_cpu", action="store_true", help="force training on CPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    # redirect the output to a log file
    parser.add_argument("--log", type=str, default=None, help="path/to/logfile.log")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.log:
        logging.getLogger().addHandler(logging.FileHandler(args.log))

    main(args)