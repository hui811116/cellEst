import csv
import io
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image

SUPPORTED_DATA_FILE_EXTENSIONS = (".csv", ".json")
_URI_COLUMN_NAMES = ("url", "uri")


def check_data_file_extension(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_DATA_FILE_EXTENSIONS:
        raise ValueError(
            f"Unsupported data file extension '{path.suffix}' for {path}; "
            f"supported: {', '.join(SUPPORTED_DATA_FILE_EXTENSIONS)}"
        )


def _check_header(columns, path: Path) -> None:
    if len(columns) < 2 or str(columns[0]).strip().lower() not in _URI_COLUMN_NAMES:
        raise ValueError(
            f"Data file {path} must have the first column 'url' or 'uri' and the second column the class name."
        )


def read_data_file(path: Path) -> list[tuple[str, str]]:
    """Return (uri, class_name) rows; the first column is url/uri, the second the class name."""
    path = Path(path)
    check_data_file_extension(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Data file is empty: {path}")
            _check_header(header, path)
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if not any(cell.strip() for cell in row):
                    continue  # blank line
                if len(row) < 2:
                    raise ValueError(
                        f"Data file {path}, line {line_number}: row is missing the class column: {row}"
                    )
                rows.append((row[0].strip(), row[1].strip()))
    else:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
            raise ValueError(f"JSON data file {path} must be a non-empty list of objects.")
        _check_header(list(records[0].keys()), path)
        rows = []
        for record in records:
            values = list(record.values())
            if list(record.keys())[:1] != list(records[0].keys())[:1] or len(values) < 2:
                raise ValueError(f"Inconsistent record in {path}: {record}")
            if not all(isinstance(value, str) for value in values[:2]):
                raise ValueError(
                    f"Inconsistent record in {path}: url/uri and class must be non-null strings: {record}"
                )
            rows.append((values[0].strip(), values[1].strip()))
    if not rows:
        raise ValueError(f"Data file has no samples: {path}")
    if any(not uri or not label for uri, label in rows):
        raise ValueError(f"Data file {path} contains empty url or class entries.")
    return rows


def _load_image(uri: str):
    if urlparse(uri).scheme in ("http", "https", "ftp", "file"):
        with urlopen(uri) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    with open(uri, "rb") as handle:
        return Image.open(handle).convert("RGB")


class ImageTableSource:
    """ImageFolder-like source (samples/targets/class_to_idx) backed by a .csv/.json file."""

    def __init__(
        self,
        data_file: Path,
        class_to_idx: dict[str, int] | None = None,
        dataset_base: Path | None = None,
    ):
        data_file = Path(data_file)
        base = Path(dataset_base).expanduser() if dataset_base is not None else data_file.parent
        rows = read_data_file(data_file)
        classes = sorted({label for _, label in rows})
        if class_to_idx is None:
            class_to_idx = {name: index for index, name in enumerate(classes)}
        elif set(classes) - set(class_to_idx):
            raise ValueError(f"{data_file} contains classes not present in the training data.")
        self.class_to_idx = class_to_idx
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.samples = [(self._resolve(uri, base), class_to_idx[label]) for uri, label in rows]
        self.targets = [target for _, target in self.samples]

    @staticmethod
    def _resolve(uri: str, base: Path) -> str:
        if urlparse(uri).scheme in ("http", "https", "ftp", "file"):
            return uri
        path = Path(uri).expanduser()
        # the base is the absoulte path, do not resolve again
        return str(path if path.is_absolute() else (base / path).resolve())

    def loader(self, uri: str):
        return _load_image(uri)

    def __len__(self):
        return len(self.samples)
