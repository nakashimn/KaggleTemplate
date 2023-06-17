import os
import glob
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
import traceback


class Preprocessor(ABC):
    def train_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()

    def test_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()

    def pred_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()

class DataPreprocessor:
    def __init__(self, config: dict[str: Any]) -> None:
        self.config = config

    def train_dataset(self) -> pd.DataFrame:
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df_meta["label_id"] = self._label_to_id(df_meta)
        return df_meta

    def _label_to_id(self, df: pd.DataFrame) -> list[str | int | float]:
        return [self.config["labels"].index(l) for l in df["label"]]

    def test_dataset(self) -> pd.DataFrame:
        df_meta = pd.read_csv(self.config["path"]["testmeta"])
        df_meta = self._cleansing(df_meta)
        return df_meta

    def pred_dataset(self) -> pd.DataFrame:
        filepaths = glob.glob(
            f"{self.config['path']['preddata']}/**/*.npz", recursive=True
        )
        df = pd.DataFrame(filepaths, columns=["filepath"])
        return df

    def _cleansing(self, df) -> pd.DataFrame:
        return df
