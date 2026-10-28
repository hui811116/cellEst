from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

from ._tabular import ImageTableSource


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


class ImageTableView(ImageFolderView):
    """View a subset of a .csv/.json listing (first column url/uri, second column class name).

    `dataset` is an ImageTableSource; `indices` defaults to every row.
    """

    def __init__(self, dataset, indices=None, transform=None):
        if not isinstance(dataset, ImageTableSource):
            raise TypeError("ImageTableView requires an ImageTableSource; use ImageTableView.from_file().")
        super().__init__(dataset, list(range(len(dataset))) if indices is None else indices, transform)

    @classmethod
    def from_file(cls, data_file, transform=None, class_to_idx=None, indices=None, dataset_base=None):
        return cls(ImageTableSource(data_file, class_to_idx, dataset_base), indices, transform)

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]


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
    use_table = data_cfg.train_data_file is not None
    if use_table:
        View = ImageTableView
        train_source = ImageTableSource(data_cfg.train_data_file, dataset_base=data_cfg.dataset_base_path)

        def load(path, _train=train_source):
            return ImageTableSource(path, _train.class_to_idx, data_cfg.dataset_base_path)

        validation_path, test_path = data_cfg.validation_data_file, data_cfg.test_data_file
    else:
        View = ImageFolderView
        train_source = ImageFolder(str(data_cfg.train_data_path))

        def load(path, _train=train_source):
            source = ImageFolder(str(path))
            if source.class_to_idx != _train.class_to_idx:
                raise ValueError("Training and evaluation datasets must have matching class folders.")
            return source

        validation_path, test_path = data_cfg.validation_data_path, data_cfg.test_data_path

    all_train_indices = list(range(len(train_source)))
    if validation_path is None:
        train_indices, validation_indices = _split_indices(
            all_train_indices, train_source.targets, data_cfg.split_ratio
        )
        validation_source = train_source
    else:
        train_indices = all_train_indices
        validation_source = load(validation_path)
        validation_indices = list(range(len(validation_source)))

    train_dataset = View(train_source, train_indices, transform_train)
    validation_dataset = View(validation_source, validation_indices, transform_test)
    test_dataset = None
    if test_path is not None:
        test_source = load(test_path)
        test_dataset = View(test_source, list(range(len(test_source))), transform_test)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )
