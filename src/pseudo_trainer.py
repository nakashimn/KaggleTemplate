import os
import shutil
import importlib
import argparse
import random
import subprocess
import pathlib
import glob
import datetime
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import traceback

from components.preprocessor import DataPreprocessor
from components.datamodule import NlpDatasetPseudo, DataModulePseudo
from components.models import NlpModelPseudo
from components.validations import MinLoss, ValidResult, ConfusionMatrix, F1Score, LogLoss

class PseudoTrainer:
    def __init__(
        self, Model, DataModule, Dataset, Tokenizer, ValidResult, MinLoss,
        df_train, df_pseudo, config, transforms, mlflow_logger
    ):
        # const
        self.mlflow_logger = mlflow_logger
        self.config = config
        self.df_train = df_train
        self.df_pseudo = df_pseudo
        self.transforms = transforms
        self.kfold = eval(self.config["kfold"]["name"])(
            **self.config["kfold"]["params"]
        )

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer
        self.MinLoss = MinLoss
        self.ValidResult = ValidResult

        # variable
        self.min_loss = self.MinLoss()
        self.val_probs = self.ValidResult()
        self.val_labels = self.ValidResult()

    def run(self):
        iterator_kfold = enumerate(
            self.kfold.split(
                self.df_train,
                self.df_train[self.config["label"]],
                self.df_train[self.config["group"]]
            )
        )
        for fold, (idx_train, idx_val) in iterator_kfold:
            # create datamodule
            datamodule = self._create_pseudo_datamodule(idx_train, idx_val)

            # train crossvalid models
            min_loss = self._train_with_crossvalid(datamodule, fold)
            self.min_loss.update(min_loss)

            # valid
            val_probs, val_labels = self._valid(datamodule, fold)
            self.val_probs.append(val_probs)
            self.val_labels.append(val_labels)

        # log
        self.mlflow_logger.log_metrics({"train_min_loss": self.min_loss.value})

        # train final model
        if self.config["train_with_alldata"]:
            datamodule = self._create_datamodule_with_alldata()
            self._train_without_valid(datamodule, self.min_loss.value)

    def _create_pseudo_datamodule(self, idx_train, idx_val):
        df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
        df_val_fold = self.df_train.loc[idx_val].reset_index(drop=True)
        datamodule = self.DataModule(
            df_train=df_train_fold,
            df_val=df_val_fold,
            df_pred=self.df_pseudo.drop(self.config["label"], axis=1),
            Dataset=self.Dataset,
            Tokenizer=self.Tokenizer,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _train_with_crossvalid(self, datamodule, fold):
        model = self.Model(self.config["model"])
        checkpoint_name = f"best_loss_{fold}"

        earystopping = EarlyStopping(
            **self.config["earlystopping"]
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            **self.config["checkpoint"]
        )

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )
        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

        df_train = datamodule.df_train.drop("essay", axis=1)
        df_train.to_csv(f"{self.config['path']['temporal_dir']}/train_{fold}.csv", index=None)
        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/train_{fold}.csv"
        )

        min_loss = model.min_loss
        return min_loss

    def _valid(self, datamodule, fold):
        checkpoint_name = f"best_loss_{fold}"
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt",
            config=self.config["model"]
        )
        model.eval()

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            **self.config["trainer"]
        )

        trainer.validate(model, datamodule=datamodule)

        val_probs = model.val_probs
        val_labels = model.val_labels
        return val_probs, val_labels

def create_mlflow_logger(config):
    timestamp = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y/%m/%d %H:%M:%S"
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=timestamp
    )
    return mlflow_logger

def update_model(config, filepath_config):
    filepaths_ckpt = glob.glob(f"{config['path']['temporal_dir']}/*.ckpt")
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    for filepath_ckpt in filepaths_ckpt:
        filename = pathlib.Path(filepath_ckpt).name
        shutil.move(filepath_ckpt, str(dirpath_model / filename))
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))

def upload_model(config, message):
    try:
        subprocess.run(
            ["kaggle", "datasets", "version", "-m", message],
            cwd=config["path"]["model_dir"]
        )
    except:
        print(traceback.format_exc())

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="stem of config filepath.",
        type=str,
        required=True
    )
    parser.add_argument(
        "-m", "--message",
        help="message for upload to kaggle datasets.",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__=="__main__":

    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # logger
    mlflow_logger = create_mlflow_logger(config)
    mlflow_logger.log_hyperparams(config)
    mlflow_logger.experiment.log_artifact(
        mlflow_logger.run_id,
        f"./config/{args.config}.py"
    )

    # Preprocess
    data_preprocessor = DataPreprocessor(config)
    fix_seed(config["random_seed"])
    df_train = data_preprocessor.train_dataset()
    df_pseudo = data_preprocessor.pseudo_dataset()

    # Training
    trainer = PseudoTrainer(
        NlpModelPseudo,
        DataModulePseudo,
        NlpDatasetPseudo,
        AutoTokenizer,
        ValidResult,
        MinLoss,
        df_train,
        df_pseudo,
        config,
        None,
        mlflow_logger
    )
    trainer.run()

    # Validation Result

    # Update model
    update_model(config, f"./config/{args.config}.py")
    upload_model(config, args.message)
