import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import traceback

class NlpDataset(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = self.read_values(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self.read_labels(df)
        self.tokenizer = Tokenizer.from_pretrained(
            self.config["base_model_name"],
            use_fast=self.config["use_fast_tokenizer"]
        )
        self.transform = transform

    def __len__(self):
        return len(self.val)

    def __getitem__(self, idx):
        ids, masks = self.tokenize(self.val[idx])
        if self.transform is not None:
            ids = self.transform(ids)
        if self.labels is not None:
            labels = self.labels[idx]
            return ids, masks, labels
        return ids, masks

    def read_values(self, df):
        values = df["features"].values
        return values

    def read_labels(self, df):
        labels = F.one_hot(
            torch.tensor(
                [self.config[self.config["label"]][d] for d in df[self.config["label"]]]
            ),
            num_classes=self.config["num_class"]
        ).float()
        return labels

    def tokenize(self, text):
        token = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding="max_length"
        )
        ids = torch.tensor(token["input_ids"], dtype=torch.long)
        masks = torch.tensor(token["attention_mask"], dtype=torch.long)
        return ids, masks



class DataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        df_val,
        df_pred,
        Dataset,
        Tokenizer,
        config,
        transforms
    ):
        super().__init__()

        # const
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.transforms = self.read_transforms(transforms)

        # class
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

    def read_transforms(self, transforms):
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self):
        dataset = self.Dataset(
            self.df_train,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["train"]
        )
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(
            self.df_val,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["valid"]
        )
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(
            self.df_pred,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["pred"]
        )
        return DataLoader(dataset, **self.config["pred_loader"])

class NlpDatasetPseudo(NlpDataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = self.read_values(df)
        self.labels = None
        self.pseudos = None
        if self.config["label"] in df.keys():
            self.labels = self.read_labels(df)
            self.pseudos = df["pseudo"].values
        self.tokenizer = Tokenizer.from_pretrained(
            self.config["base_model_name"],
            use_fast=self.config["use_fast_tokenizer"]
        )
        self.transform = transform

    def __getitem__(self, idx):
        ids, masks = self.tokenize(self.val[idx])
        if self.transform is not None:
            ids = self.transform(ids)
        if self.labels is not None:
            labels = self.labels[idx]
            pseudos = self.pseudos[idx]
            return ids, masks, labels, pseudos
        return ids, masks

class DataModulePseudo(DataModule):
    def __init__(
        self,
        df_train,
        df_val,
        df_pred,
        Dataset,
        Tokenizer,
        config,
        transforms
    ):
        super().__init__(
            df_train,
            df_val,
            df_pred,
            Dataset,
            Tokenizer,
            config,
            transforms
        )
        # const
        self.df_train_org = None
        if df_train is not None:
            self.df_train_org = df_train.copy()

        # variables
        self.df_train = df_train        # updated with pseudo labeled data

    def update_train_dataloader(self, pseudo_label_probs):
        idx_conf_pseudo_labels, conf_pseudo_labels = \
            self.pickup_confidential_pseudo_labels(pseudo_label_probs)
        df_pseudo_train = self.df_pred.loc[idx_conf_pseudo_labels].reset_index(drop=True)
        df_pseudo_train[self.config["dataset"]["label"]] = \
            [self.config["dataset"]["labels"][i] for i in conf_pseudo_labels]
        self.df_train = pd.concat([self.df_train_org, df_pseudo_train]).reset_index(drop=True)
        confidential_pseudo_rate = np.sum(idx_conf_pseudo_labels)/len(pseudo_label_probs)
        return confidential_pseudo_rate

    def pickup_confidential_pseudo_labels(self, pseudo_label_probs):
        idx_confidential_pseudo_labels = (
            np.max(pseudo_label_probs, axis=1)>=self.config["pseudo_confidential_threshold"]
        )
        confidential_pseudo_labels = np.argmax(pseudo_label_probs, axis=1)[idx_confidential_pseudo_labels]
        return idx_confidential_pseudo_labels, confidential_pseudo_labels
