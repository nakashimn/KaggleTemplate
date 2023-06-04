import numpy as np
import pandas as pd
import librosa
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Tv
import albumentations as A
from pytorch_lightning import LightningDataModule
import traceback

################################################################################
# For EfficientNetBaseModel
################################################################################
class ImgDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.pre_transform = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor = Tv.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        melspec = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        if self.transform is not None:
            melspec = self.transform(melspec)
        if self.labels is not None:
            labels = self.labels[idx]
            return melspec, labels
        return melspec

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _read_melspec(self, filepath):
        melspec = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    def _normalize(self, melspec, eps=1e-6):
        melspec = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

    def _read_labels(self, df):
        labels = torch.tensor(df.apply(self._to_onehot), dtype=torch.float32)
        return labels

    def _to_onehot(self, series):
        return [1 if l in series else 0 for l in self.config["labels"]]

################################################################################
# DataModule
################################################################################
class DataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_pred, Dataset, config, transforms):
        super().__init__()

        # const
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.transforms = self._read_transforms(transforms)

        # class
        self.Dataset = Dataset

    def _read_transforms(self, transforms):
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self):
        if (self.df_train is None) or (len(self.df_train) == 0):
            return None
        dataset = self.Dataset(
            self.df_train, self.config["dataset"], self.transforms["train"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            **self.config["dataloader"]
        )
        return dataloader

    def val_dataloader(self):
        if (self.df_val is None) or (len(self.df_val) == 0):
            return None
        dataset = self.Dataset(
            self.df_val, self.config["dataset"], self.transforms["valid"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader

    def predict_dataloader(self):
        if (self.df_pred is None) or (len(self.df_pred) == 0):
            return None
        dataset = self.Dataset(
            self.df_pred, self.config["dataset"], self.transforms["pred"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader
