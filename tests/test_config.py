import csv
import io
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import DataLoader
from torchvision import transforms

from cellest.config import DataConfig
from cellest.dataloader import build_datasets
from cellest.dataloader import _tabular

CLASSES = ("alpha", "beta")
TRANSFORM = transforms.ToTensor()
rng = np.random.default_rng(0)


def _random_image(size=8):
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def _png_bytes():
    buffer = io.BytesIO()
    _random_image().save(buffer, format="PNG")
    return buffer.getvalue()


def _write_table(path, rows, header=("url", "class")):
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    else:
        path.write_text(json.dumps([dict(zip(header, row)) for row in rows]))
    return path


@pytest.fixture
def images(tmp_path):
    """Random images on disk: 6 per class, returned as (relative name, class) rows."""
    rows = []
    for label in CLASSES:
        for i in range(6):
            name = f"{label}_{i}.png"
            _random_image().save(tmp_path / name)
            rows.append((name, label))
    return rows


@pytest.fixture
def image_folders(tmp_path):
    root = tmp_path / "folders"
    for split in ("train", "test"):
        for label in CLASSES:
            (root / split / label).mkdir(parents=True)
            for i in range(6):
                _random_image().save(root / split / label / f"{i}.png")
    return root


# ---------- DataConfig: path variant ----------


def test_path_config_valid(image_folders):
    cfg = DataConfig(train_data_path=image_folders / "train", test_data_path=image_folders / "test")
    assert cfg.train_data_file is None
    assert cfg.split_ratio == 0.8


def test_path_config_aliases(image_folders):
    cfg = DataConfig(train_path=image_folders / "train", test_path=image_folders / "test")
    assert cfg.test_data_path == image_folders / "test"


def test_path_config_missing_path(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        DataConfig(train_data_path=tmp_path / "nope")


def test_path_config_missing_test_path(image_folders, tmp_path):
    with pytest.raises(ValidationError, match="Test data path"):
        DataConfig(train_data_path=image_folders / "train", test_data_path=tmp_path / "nope")


def test_requires_a_train_source():
    with pytest.raises(ValidationError, match="exactly one"):
        DataConfig()


def test_rejects_both_train_sources(image_folders, images, tmp_path):
    table = _write_table(tmp_path / "d.csv", images)
    with pytest.raises(ValidationError, match="exactly one"):
        DataConfig(train_data_path=image_folders / "train", train_data_file=table)


def test_invalid_split_ratio(image_folders):
    with pytest.raises(ValidationError):
        DataConfig(train_data_path=image_folders / "train", split_ratio=1.5)


# ---------- DataConfig: file variant ----------


@pytest.mark.parametrize("ext", ["csv", "json"])
def test_file_config_valid(tmp_path, images, ext):
    train = _write_table(tmp_path / f"train.{ext}", images)
    test = _write_table(tmp_path / f"test.{ext}", images)
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=train, test_data_file=test)
    assert cfg.train_data_path is None
    assert cfg.test_data_file == test


def test_file_config_uri_header_accepted(tmp_path, images):
    train = _write_table(tmp_path / "train.csv", images, header=("URI", "label"))
    DataConfig(dataset_base_path=tmp_path, train_data_file=train)


def test_file_config_unsupported_extension(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("url,class\n")
    with pytest.raises(ValidationError, match="Unsupported data file extension"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_file_config_unsupported_extension_on_test_file(tmp_path, images):
    train = _write_table(tmp_path / "train.csv", images)
    bad = tmp_path / "test.tsv"
    bad.write_text("x")
    with pytest.raises(ValidationError, match="Unsupported data file extension"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=train, test_data_file=bad)


def test_file_config_missing_file(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=tmp_path / "missing.csv")


@pytest.mark.parametrize("ext", ["csv", "json"])
def test_file_config_bad_first_column(tmp_path, images, ext):
    train = _write_table(tmp_path / f"train.{ext}", images, header=("class", "url"))
    with pytest.raises(ValidationError, match="first column"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=train)


def test_file_config_single_column(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("url\na.png\n")
    with pytest.raises(ValidationError, match="first column"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_file_config_empty_file(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("")
    with pytest.raises(ValidationError, match="empty"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_file_config_header_only(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("url,class\n")
    with pytest.raises(ValidationError, match="no samples"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_file_config_empty_class(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("url,class\na.png,\n")
    with pytest.raises(ValidationError, match="empty url or class"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_json_must_be_list_of_objects(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"url": "a.png"}))
    with pytest.raises(ValidationError, match="list of objects"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=path)


def test_file_and_path_cannot_be_mixed(tmp_path, images, image_folders):
    train = _write_table(tmp_path / "train.csv", images)
    with pytest.raises(ValidationError, match="cannot be mixed"):
        DataConfig(dataset_base_path=tmp_path, train_data_file=train, test_data_path=image_folders / "test")
    with pytest.raises(ValidationError, match="cannot be mixed"):
        DataConfig(train_data_path=image_folders / "train", test_data_file=train)


# ---------- build_datasets ----------


def _check_loader(dataset, expected_len):
    assert len(dataset) == expected_len
    total = 0
    for batch, targets in DataLoader(dataset, batch_size=4):
        assert batch.shape[1:] == (3, 8, 8)
        assert set(targets.tolist()) <= {0, 1}
        total += len(targets)
    assert total == expected_len


def test_build_datasets_from_folders(image_folders):
    cfg = DataConfig(train_data_path=image_folders / "train", test_data_path=image_folders / "test")
    train, val, test = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert len(train) + len(val) == 12
    _check_loader(train, len(train))
    _check_loader(test, 12)


@pytest.mark.parametrize("ext", ["csv", "json"])
def test_build_datasets_from_files_with_split(tmp_path, images, ext):
    train_file = _write_table(tmp_path / f"train.{ext}", images)
    test_file = _write_table(tmp_path / f"test.{ext}", images[:4] + images[6:10])
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=train_file, test_data_file=test_file, split_ratio=0.75)
    train, val, test = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert (len(train), len(val), len(test)) == (9, 3, 8)
    assert not set(train.indices) & set(val.indices)
    _check_loader(train, 9)
    _check_loader(val, 3)
    _check_loader(test, 8)


def test_build_datasets_without_test_file(tmp_path, images):
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=_write_table(tmp_path / "train.csv", images))
    _, _, test = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert test is None


def test_build_datasets_explicit_validation_file(tmp_path, images):
    train_file = _write_table(tmp_path / "train.csv", images)
    val_file = _write_table(tmp_path / "val.json", images[:2] + images[6:8])
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=train_file, validation_data_file=val_file)
    train, val, _ = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert (len(train), len(val)) == (12, 4)
    assert val.dataset.class_to_idx == train.dataset.class_to_idx


def test_class_indices_follow_training_file(tmp_path, images):
    train_file = _write_table(tmp_path / "train.csv", images)
    # the test file lists only "beta"; it must still map to the training index 1
    test_file = _write_table(tmp_path / "test.csv", [r for r in images if r[1] == "beta"])
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=train_file, test_data_file=test_file)
    _, _, test = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert {target for _, target in test.dataset.samples} == {1}


def test_unknown_class_in_test_file_raises(tmp_path, images):
    train_file = _write_table(tmp_path / "train.csv", images)
    test_file = _write_table(tmp_path / "test.csv", [(images[0][0], "gamma")])
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=train_file, test_data_file=test_file)
    with pytest.raises(ValueError, match="not present in the training data"):
        build_datasets(cfg, TRANSFORM, TRANSFORM)


def test_relative_and_absolute_paths(tmp_path, images):
    rows = [(str(tmp_path / images[0][0]), images[0][1])] + images[1:]
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=_write_table(tmp_path / "sub.csv", rows))
    train, val, _ = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert len(train) + len(val) == 12


def test_in_memory_urls(tmp_path, monkeypatch):
    payload = _png_bytes()
    requested = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    def fake_urlopen(url):
        requested.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr(_tabular, "urlopen", fake_urlopen)
    rows = [(f"https://example.test/{label}/{i}.png", label) for label in CLASSES for i in range(4)]
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=_write_table(tmp_path / "urls.json", rows))
    train, _, _ = build_datasets(cfg, TRANSFORM, TRANSFORM)
    _check_loader(train, len(train))
    assert requested and all(url.startswith("https://example.test/") for url in requested)


def test_split_too_small_raises(tmp_path):
    cfg = DataConfig(dataset_base_path=tmp_path, train_data_file=_write_table(tmp_path / "train.csv", [("a.png", "alpha")]))
    with pytest.raises(ValueError, match="At least two"):
        build_datasets(cfg, TRANSFORM, TRANSFORM)


def test_image_table_view_from_file(tmp_path, images):
    from cellest.dataloader import ImageTableView

    view = ImageTableView.from_file(_write_table(tmp_path / "d.json", images), TRANSFORM, dataset_base=tmp_path)
    assert len(view) == 12
    assert view.classes == list(CLASSES)
    assert view.class_to_idx == {"alpha": 0, "beta": 1}
    assert view.targets == [0] * 6 + [1] * 6
    image, target = view[7]
    assert image.shape == (3, 8, 8) and target == 1


def test_image_table_view_rejects_plain_dataset():
    from cellest.dataloader import ImageTableView

    with pytest.raises(TypeError):
        ImageTableView(object())


def test_dataset_base_joins_relative_uris(tmp_path, images):
    base = tmp_path / "base"
    base.mkdir()
    for name, _ in images:
        (tmp_path / name).rename(base / name)
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    cfg = DataConfig(
        train_data_file=_write_table(table_dir / "train.csv", images),
        test_data_file=_write_table(table_dir / "test.json", images),
        dataset_base_path=base,
    )
    train, val, test = build_datasets(cfg, TRANSFORM, TRANSFORM)
    assert train.dataset.samples[0][0] == str((base / images[0][0]).resolve())
    _check_loader(test, 12)


def test_dataset_base_validation(tmp_path, images, image_folders):
    table = _write_table(tmp_path / "train.csv", images)
    with pytest.raises(ValidationError, match="not a directory"):
        DataConfig(train_data_file=table, dataset_base_path=tmp_path / "nope")
    with pytest.raises(ValidationError, match="only applies"):
        DataConfig(train_data_path=image_folders / "train", dataset_base_path=tmp_path)


def test_from_yaml_resolves_base_against_config_dir(tmp_path, images, monkeypatch):
    from cellest.config import CellestClaTrainConfig

    cfg_dir = tmp_path / "conf"
    cfg_dir.mkdir()
    _write_table(tmp_path / "train.csv", images)
    (cfg_dir / "config.yaml").write_text(
        "data:\n  dataset_base_path: ..\n  train_data_file: train.csv\n"
        "model:\n  family: cnn\n  architecture: resnet101\n"
        "train:\n  batch_size: 2\n  epochs: 1\n  learning_rate: 0.1\n  optimizer: adam\n  output_dir: out\n"
    )
    monkeypatch.chdir("/")  # the working directory must not matter
    cfg = CellestClaTrainConfig.from_yaml(cfg_dir / "config.yaml")
    assert cfg.data.dataset_base_path == tmp_path.resolve()
    assert cfg.data.train_data_file == tmp_path.resolve() / "train.csv"
    train, val, _ = build_datasets(cfg.data, TRANSFORM, TRANSFORM)
    assert len(train) + len(val) == 12
