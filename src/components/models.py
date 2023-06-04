import os
import sys
import pathlib
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule
import timm
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from augmentations import Mixup, LabelSmoothing

################################################################################
# EfficientNet
################################################################################
class EfficientNetModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.bn, self.encoder, self.fc = self._create_model()
        self.criterion = eval(config["loss"]["name"])(**self.config["loss"]["params"])

        # augmentation
        self.mixup = Mixup(config["mixup"]["alpha"])
        self.label_smoothing = LabelSmoothing(
            config["label_smoothing"]["eps"], config["num_class"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        # batch_normalization
        bn = nn.BatchNorm2d(self.config["n_mels"])
        # basemodel
        base_model = timm.create_model(
            self.config["base_model_name"],
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=1
        )
        layers = list(base_model.children())[:-2]
        encoder = nn.Sequential(*layers)
        # linear
        fc = nn.Sequential(
            nn.Linear(
                encoder[-1].num_features * 10,
                self.config["fc_mid_dim"],
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                self.config["fc_mid_dim"],
                self.config["num_class"],
                bias=True
            )
        )
        return bn, encoder, fc

    def forward(self, input_data):
        x = input_data[:, [0], :, :]
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.mean(dim=2)
        x = x.flatten(start_dim=1)
        out = self.fc(x)
        return out

    def training_step(self, batch, batch_idx):
        img, labels = batch
        img, labels = self.mixup(img, labels)
        labels = self.label_smoothing(labels)
        logits = self.forward(img)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        logits = self.forward(img)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        img = batch
        logits = self.forward(img)
        prob = logits.softmax(axis=1).detach()
        return {"prob": prob}

    def training_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)

        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.log(f"val_loss", metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]
