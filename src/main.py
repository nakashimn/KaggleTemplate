import os
import shutil
import importlib
import argparse
from argparse import Namespace
import pathlib
import glob
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, LightningDataModule
from typing import Any
import traceback

from components.utility import print_info, fix_seed
from components.preprocessor import DataPreprocessor
from components.augmentations import Augmentation
from components.trainer import Trainer


def update_config(config: dict[str, Any], filepath_config: str) -> None:
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))


def remove_exist_models(config: dict[str, Any]) -> None:
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)


def import_classes(
    config: dict[str, Any]
) -> tuple[LightningModule, Dataset, LightningDataModule, Augmentation]:
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"), config["model"]["ClassName"]
    )
    Dataset = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["dataset"]["ClassName"],
    )
    DataModule = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["ClassName"],
    )
    Augmentation = getattr(
        importlib.import_module(f"components.augmentations"),
        config["augmentation"]["ClassName"],
    )
    # debug message
    print_info(
        {
            "Components": {
                "Model": Model.__name__,
                "Dataset": Dataset.__name__,
                "DataModule": DataModule.__name__,
                "Augmentation": Augmentation.__name__,
            }
        }
    )
    return Model, Dataset, DataModule, Augmentation


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="stem of config filepath.", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config
    from config.sample import config

    # import Classes
    Model, Dataset, DataModule, Augmentation = import_classes(config)

    # update config
    update_config(config, f"./config/{args.config}.py")

    # create model dir
    os.makedirs(config["path"]["model_dir"], exist_ok=True)

    # torch setting
    torch.set_float32_matmul_precision("medium")

    # Preprocess
    remove_exist_models(config)
    fix_seed(config["random_seed"])
    data_preprocessor = DataPreprocessor(config)
    df_train = data_preprocessor.train_dataset()

    # Training
    trainer = Trainer(Model, DataModule, Dataset, Augmentation, df_train, config)
    trainer.run()
