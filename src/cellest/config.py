from pathlib import Path
from typing import Literal
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from cellest.classifiers._macro import SUPPORTED_ARCHITECTURES, CLASSIFIER_TYPES
"""
Config.yaml format:
        data:
            train_path: path/to/train/dataset
            test_path: path/to/test/dataset
            # if train_path and test_path are not specified, then the dataset will be split into train and test sets
            split_ratio: 0.8 # must be specified if train_path and test_path are not specified
        model:
            arch: [cell|pretrained]
            # if arch is cell, then the model will be trained from scratch
            premodel: [resnet|inception] # must be specified if arch is cell
            # if arch is pretrained, then the model will be finetuned from a pretrained model
            pretrained_path: [example: facebook/dino:resnet50] # must be specified if archi is pretrained
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
