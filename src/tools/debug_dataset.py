import os
import random
import sys
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from transformers import AutoTokenizer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.preprocessor import DataPreprocessor
from components.datamodule import NlpDataset

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###
# sample
###

data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()

# FpDataSet
fix_seed(config["random_seed"])
dataset = NlpDataset(df_train, config["datamodule"]["dataset"], AutoTokenizer)
batch = dataset.__getitem__(0)
dataset.val
