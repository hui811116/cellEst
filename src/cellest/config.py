from pathlib import Path
from typing import Literal
import json
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from cellest.classifiers.cell._constants import SUPPORTED_ARCHITECTURES as CELL_SUPPORTED_ARCHITECTURES
"""
Config.yaml format:
        data:
            train_path: path/to/train/dataset
            test_path: path/to/test/dataset
            # if train_path and test_path are not specified, then the dataset will be split into train and test sets
            split_ratio: 0.8 # must be specified if train_path and test_path are not specified
        model:
            family: [cnn|transformer]
            architecture: [resnet101|inception_v3|convnext_base|efficientnet_v2_m|vision_transformer]
            pretrained_path: [example: facebook/dino-vitb16] # required for transformers
            # for all cases, must be specified
            classifier_type: [mlp|linear]
            model_nickname: [example: mark2] # optional, random string+timestamp if not specified 
        train:
            batch_size: 32
            epochs: 100
            learning_rate: 0.001
            weight_decay: 0.0001
            optimizer: [adam|sgd]
            scheduler: [step|cosine]
            step_size: 30 # must be specified if scheduler is step
            gamma: 0.1 # must be specified if scheduler is step
"""

class InferenceDataConfig(BaseModel):
    """Dataset settings for inference."""

    model_config = ConfigDict(extra="forbid")

    data_path: Path = Field(validation_alias=AliasChoices("dataset_path", "data_path"))

    @model_validator(mode="after")
    def validate_data_config(self) -> "InferenceDataConfig":
        if not self.data_path.is_dir():
            raise ValueError(f"Inference dataset directory does not exist: {self.data_path}")
        return self


class InferenceConfig(BaseModel):
    """Runtime settings for a GradCAM inference run."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    data: InferenceDataConfig
    output_dir: Path = Path("gradcam_output")
    target_class: str
    batch_size: int = Field(default=1, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_model_path(self) -> "InferenceConfig":
        if not self.model_path.is_file():
            raise ValueError(f"Model artifact does not exist: {self.model_path}")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "InferenceConfig":
        yaml_path = yaml_path.resolve()
        base_dir = Path.cwd()
        with yaml_path.open(encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file) or {}

        if "model_path" in config_data:
            config_data["model_path"] = _resolve_path(
                config_data["model_path"], base_dir
            )
        config_data["output_dir"] = _resolve_path(
            config_data.get("output_dir", "gradcam_output"), base_dir
        )
        data_config = config_data.get("data", {})
        for data_key in ("dataset_path", "data_path"):
            if data_key in data_config:
                data_config[data_key] = _resolve_path(
                    data_config[data_key], base_dir
                )
        return cls.model_validate(config_data)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


    

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_data_path: Path = Field(
        validation_alias=AliasChoices("train_data_path", "train_path", "data_path")
    )
    validation_data_path: Path | None = None
    test_data_path: Path | None = Field(
        default=None, validation_alias=AliasChoices("test_data_path", "test_path")
    )
    split_ratio: float = Field(default=0.8, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_data_config(self) -> "DataConfig":
        # traing datat path is required
        if not self.train_data_path.exists():
            raise ValueError(f"Train data path does not exist: {self.train_data_path}")
        # validation split is optional, if presented, split the training data only
        if self.validation_data_path is not None and not self.validation_data_path.exists():
            raise ValueError(f"Validation data path does not exist: {self.validation_data_path}")
        # split ratio is only used if validation data path is not provided
        if self.validation_data_path is None and not (0 < self.split_ratio < 1):
            raise ValueError(f"Split ratio must be between 0 and 1: {self.split_ratio}")
        # test path is optional, if presented, check if it exists
        if self.test_data_path is not None and not self.test_data_path.exists():
            raise ValueError(f"Test data path does not exist: {self.test_data_path}")
        return self

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["cnn", "transformer"]
    architecture: str
    pretrained_path: str | None = None
    classifier_type: Literal["mlp", "linear"] = "linear"
    model_nickname: str | None = None

    @model_validator(mode="after")
    def validate_model_config(self) -> "ModelConfig":
        if self.family == "cnn" and self.architecture not in CELL_SUPPORTED_ARCHITECTURES:
            raise ValueError("CNN architecture must be one of the supported architectures.")
        if self.family == "transformer" and self.pretrained_path is None:
            raise ValueError("Transformer models require pretrained_path.")
        if self.family == "transformer" and self.architecture != "vision_transformer":
            raise ValueError("Transformer architecture must be 'vision_transformer'.")

        return self


class InferenceModelConfig(BaseModel):
    """Model metadata loaded from a training run's summary.json."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    num_classes: int = Field(gt=1)
    class_names: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_class_metadata(self) -> "InferenceModelConfig":
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match the number of class_names.")
        return self

    @classmethod
    def from_summary(cls, summary_path: Path) -> "InferenceModelConfig":
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        return cls.model_validate(
            {
                "model": summary["model"],
                "num_classes": summary["num_classes"],
                "class_names": summary["class_names"],
            }
        )

class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    scheduler: Literal["step", "cosine", "none"] = "none"
    step_size: int | None = Field(default=None, gt=0)
    gamma: float | None = Field(default=None, gt=0, lt=1)
    output_dir: Path = Path("results")

    @model_validator(mode="after")
    def validate_train_config(self) -> "TrainConfig":
        if self.scheduler == "step" and (self.step_size is None or self.gamma is None):
            raise ValueError("Specify step_size and gamma for step scheduler.")

        if self.scheduler == "cosine" and self.gamma is not None:
            raise ValueError("Gamma should not be specified for cosine scheduler.")

        return self

class CellestClaTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CellestClaTrainConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
