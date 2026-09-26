from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from cellest.classifiers._macro import SUPPORTED_ARCHITECTURES, CLASSIFIER_TYPES

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path: Path | None = None
    train_path: Path | None = None
    test_path: Path | None = None
    split_ratio: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_data_source(self) -> "DataConfig":
        has_explicit_paths = self.train_path is not None or self.test_path is not None

        if has_explicit_paths and (self.train_path is None or self.test_path is None):
            raise ValueError("Specify both train_path and test_path.")

        if has_explicit_paths and self.data_path is not None:
            raise ValueError("Specify either data_path with split_ratio, or train_path and test_path.")

        if not has_explicit_paths and (self.data_path is None or self.split_ratio is None):
            raise ValueError(
                "Specify data_path and split_ratio, or provide train_path and test_path."
            )

        return self

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arch: Literal[tuple(SUPPORTED_ARCHITECTURES)]
    premodel: str | None = None
    pretrained_path: str | None = None
    classifier_type: Literal[tuple(CLASSIFIER_TYPES)] = "linear"
    model_nickname: str | None = None

    @model_validator(mode="after")
    def validate_model_config(self) -> "ModelConfig":
        if self.arch == "pretrained" and self.premodel is None:
            raise ValueError("Specify premodel for pretrained architecture.")

        if self.arch == "pretrained" and self.pretrained_path is None:
            raise ValueError("Specify pretrained_path for pretrained architecture.")

        return self

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
