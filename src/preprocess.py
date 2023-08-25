from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from sklearn.preprocessing import LabelEncoder

from .tool import get_logger

logger = get_logger(__name__)


class Preprocessor:
    def __init__(self, cfg):
        self.cfg: dict = cfg
        self.agg: dict[str, Any] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}

    def common_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def aggregate(self, df: pd.DataFrame):
        self.agg = {}
        target = self.cfg["target"]
        for c in self.cfg["cat_feat"]:
            tmp = df[[c, target]].fillna({c: "nan"})
            self.agg[c] = tmp.groupby(c)[target].agg(self.cfg["stats"])

    def target_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.cfg["cat_feat"]:
            for s in self.cfg["stats"]:
                df[f"{c}_enc_{s}"] = df[c].fillna("nan").map(self.agg[c][s])
        return df

    def fit_label_encode(self, df: pd.DataFrame):
        self.label_encoders = {}
        for c in self.cfg["cat_feat"]:
            le = LabelEncoder()
            le.fit(df[c].fillna("nan"))
            self.label_encoders[c] = le

    def label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        for c, le in self.label_encoders.items():
            df[f"{c}_le"] = le.transform(df[c].fillna("nan"))
        return df

    def preprocess(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = self.common_preprocess(df)

        if is_train:
            self.aggregate(df)
            self.fit_label_encode(df)

        df = self.target_encode(df)
        df = self.label_encode(df)
        return df

    @classmethod
    def load(cls, outpath: str) -> "Preprocessor":
        with open(outpath, "rb") as f:
            return joblib.load(f)

    def save(self, outpath: str):
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "wb") as f:
            joblib.dump(self, f, compress=3)
