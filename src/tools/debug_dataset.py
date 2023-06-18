import os
import random
import sys
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.preprocessor import DataPreprocessor
from components.datamodule import ImgDataset, DataModule

###
# sample
###
# prepare input
data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()
df_test = data_preprocessor.test_dataset()
df_pred = data_preprocessor.pred_dataset()

# DataSet
try:
    dataset = ImgDataset(
        df_train,
        config["datamodule"]["dataset"]
    )
    for i in tqdm(range(dataset.__len__())):
        batch = dataset.__getitem__(i)

except:
    print(traceback.format_exc())

# DataLoader
try:
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )
    for data in tqdm(dataloader):
        print(data)
except:
    print(traceback.format_exc())

# DataModule
try:
    datamodule = DataModule(
        df_train=df_train,
        df_val=None,
        df_pred=None,
        Dataset=ImgDataset,
        config=config,
        transforms=None
    )
except:
    print(traceback.format_exc())
