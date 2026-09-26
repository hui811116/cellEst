from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split

def build_datasets(data_cfg, transform_train, transform_test):
    """resolve train/test ImageFolder datasets from explicit paths or a split ratio"""
    if data_cfg.train_path is not None:
        return (
            ImageFolder(str(data_cfg.train_path), transform=transform_train),
            ImageFolder(str(data_cfg.test_path), transform=transform_test),
        )

    full_dataset = ImageFolder(str(data_cfg.data_path))
    n_train = int(len(full_dataset) * data_cfg.split_ratio)
    n_test = len(full_dataset) - n_train
    train_indices, test_indices = random_split(full_dataset, [n_train, n_test])
    train_dataset = ImageFolder(str(data_cfg.data_path), transform=transform_train)
    test_dataset = ImageFolder(str(data_cfg.data_path), transform=transform_test)
    return (
        Subset(train_dataset, train_indices.indices),
        Subset(test_dataset, test_indices.indices),
    )