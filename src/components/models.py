import os
import sys
import pathlib
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule
import timm
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from loss_functions import FocalLoss, PseudoLoss

class ImgRecogModelBase(LightningModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model = self.create_base_model()
        self.dropout = nn.Dropout(self.config["dropout_rate"])

        self.criterion = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    @abstractmethod
    def create_base_model(self):
        pass

    @abstractmethod
    def forward(self, imgs):
        pass

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        imgs = batch
        logits = self.forward(imgs)
        prob = logits.softmax(axis=1).detach()
        return {"prob": prob}

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(),
            **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer,
            **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

class ImgRecogModel(ImgRecogModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.fc = self.create_fully_connected()

    def create_base_model(self):
        base_model = timm.create_model(
            self.config["base_model_name"]
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, imgs):
        out = self.base_model(imgs)
        out = self.dropout(out[0])
        out = self.fc(out)
        return out

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

class ImgRecogModelPseudo(ImgRecogModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.fc = self.create_fully_connected()

    def create_base_model(self):
        base_model = timm.create_model(
            self.config["base_model_name"]
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, imgs):
        out = self.base_model(imgs)
        out = self.dropout(out[0])
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        imgs, labels, pseudos = batch
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels, pseudos, self.trainer.current_epoch)
        logit = logits.detach()
        label = labels.detach()
        pseudo = pseudos.detach()
        return {"loss": loss, "logit": logit, "label": label, "pseudo": pseudo}

    def validation_step(self, batch, batch_idx):
        imgs, labels, pseudos = batch
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels, pseudos, self.trainer.current_epoch)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        pseudo = pseudos.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label, "pseudo": pseudo}

    def training_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        pseudos = torch.cat([out["pseudo"] for out in outputs])
        metrics = self.criterion(logits, labels, pseudos, self.trainer.current_epoch)
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)

        # update_pseudo_labels
        pseudo_label_probs = self.predict_pseudo_labels()
        pseudo_labeled_ratio = self.trainer.datamodule.update_train_dataloader(pseudo_label_probs)
        self.log(f"pseudo_labeled_ratio", pseudo_labeled_ratio)
        self.trainer.reset_train_dataloader()

        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        pseudos = torch.cat([out["pseudo"] for out in outputs])
        metrics = self.criterion(logits, labels, pseudos, self.trainer.current_epoch)
        self.log(f"val_loss", metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().validation_epoch_end(outputs)

    def predict_pseudo_labels(self):
        pred_dataloader = self.trainer.datamodule.predict_dataloader()
        probs = []
        self.eval()
        with torch.no_grad():
            for ids, masks in tqdm(pred_dataloader):
                logits = self.forward(ids.detach().to("cuda"), masks.detach().to("cuda"))
                probs.append(logits.softmax(axis=1).detach().cpu().numpy())
        self.train()
        return np.concatenate(probs)

class ImgRecogModelFgm(ImgRecogModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.fc = self.create_fully_connected()
        self.fgm = FGM(self, **self.config["fgm"])
        self.automatic_optimization = False

    def create_base_model(self):
        base_model = timm.create_model(
            self.config["base_model_name"]
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, imgs):
        out = self.base_model(imgs)
        out = self.dropout(out[0])
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.criterion(logits, labels)
        self.manual_backward(loss)
        self.adversalial_training(batch)
        opt.step()
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def adversalial_training(self, batch):
        imgs, labels = batch
        self.fgm.attack()
        logits_adv = self.forward(imgs)
        loss_adv = self.criterion(logits_adv, labels)
        self.manual_backward(loss_adv)
        self.fgm.restore()

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

class FGM:
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        # const
        self.emb_name = emb_name
        self.epsilon = epsilon

        # variable
        self.model = model
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.linalg.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class ImgRecogModelMeanPooling(ImgRecogModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.pooler = MeanPooling()
        self.fc = self.create_fully_connected()

    def create_base_model(self):
        base_model = timm.create_model(
            self.config["base_model_name"]
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, imgs):
        out = self.base_model(imgs, output_hidden_states=False)
        out = self.pooler(out.last_hidden_state)
        out = self.dropout(out)
        out = self.fc(out)
        return out

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
