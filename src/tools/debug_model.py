import sys
import pathlib
import torch
from transformers import AutoTokenizer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# from config.distilbert import config
from config.sample import config
from components.models import NlpModel

###
# sample
###

# prepare
tokenizer = AutoTokenizer.from_pretrained(
    config["datamodule"]["dataset"]["base_model_name"],
    use_fast=config["datamodule"]["dataset"]["use_fast_tokenizer"]
)

# tokenize
token = tokenizer.encode_plus(
    "test",
    truncation=True,
    add_special_tokens=True,
    max_length=config["datamodule"]["dataset"]["max_length"],
    padding="max_length"
)
ids = torch.tensor([token["input_ids"]])
masks = torch.tensor([token["attention_mask"]])

# model
model = NlpModel(config["model"])
model(ids, masks)
