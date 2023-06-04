import os
import glob
import ast
import numpy as np
import pandas as pd
import traceback

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def train_dataset(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df_meta["label_id"] = self._label_to_id(df_meta)
        return df_meta

    def _label_to_id(self, df):
        return [self.config["labels"].index(l) for l in df["label"]]

    def test_dataset(self):
        df_meta = pd.read_csv(self.config["path"]["testmeta"])
        df_meta = self._cleansing(df_meta)
        return df_meta

    def pred_dataset(self):
        filepaths = glob.glob(
            f"{self.config['path']['preddata']}/**/*.npz", recursive=True
        )
        df = pd.DataFrame(filepaths, columns=["filepath"])
        return df

    def _cleansing(self, df):
        return df
