from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch


class ImageFolderView(Dataset):
    """View an ImageFolder subset with a transform independent of other splits."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = self.indices[index]
        path, target = self.dataset.samples[sample_index]
        image = self.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _split_indices(indices, targets, train_ratio):
    if len(indices) < 2:
        raise ValueError("At least two training images are required for train/validation splitting.")
    labels = [targets[index] for index in indices]
    try:
        return train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=torch.initial_seed() % (2**32),
        )
    except ValueError as error:
        raise ValueError(
            "Train/validation splitting requires enough images from each class; adjust split_ratio or add data."
        ) from error


def build_datasets(data_cfg, transform_train, transform_test):
    """Build training/validation splits and an optional independent test dataset."""
    train_source = ImageFolder(str(data_cfg.train_data_path))
    all_train_indices = list(range(len(train_source)))

    if data_cfg.validation_data_path is None:
        train_indices, validation_indices = _split_indices(
            all_train_indices, train_source.targets, data_cfg.split_ratio
        )
        validation_source = train_source
    else:
        train_indices = all_train_indices
        validation_source = ImageFolder(str(data_cfg.validation_data_path))
        if validation_source.class_to_idx != train_source.class_to_idx:
            raise ValueError("Training and validation datasets must have matching class folders.")
        validation_indices = list(range(len(validation_source)))

    train_dataset = ImageFolderView(train_source, train_indices, transform_train)
    validation_dataset = ImageFolderView(validation_source, validation_indices, transform_test)
    test_dataset = None
    if data_cfg.test_data_path is not None:
        test_source = ImageFolder(str(data_cfg.test_data_path))
        if test_source.class_to_idx != train_source.class_to_idx:
            raise ValueError("Training and test datasets must have matching class folders.")
        test_dataset = ImageFolderView(
            test_source, list(range(len(test_source))), transform_test
        )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )