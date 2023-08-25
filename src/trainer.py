from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import KFold


class CVModel(metaclass=ABCMeta):
    cfg = None

    @abstractmethod
    def fit(self, train: pd.DataFrame, seed: int):
        raise NotImplementedError()

    @abstractmethod
    def predict_array(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.mean(self.predict_array(test), axis=1)


def get_kfold(cfg, train: pd.DataFrame, seed: int):
    kf = KFold(n_splits=cfg["n_splits"], random_state=seed, shuffle=True)
    return kf.split(train[cfg["target"]], train[cfg["target"]])


def get_metric_func(name) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "mae":
        return mean_absolute_error
    if name == "mse":
        return mean_squared_error
    if name == "rmse":
        return partial(mean_squared_error, squared=False)
    if name == "msle":
        return mean_squared_log_error
    if name == "mape":
        return mean_absolute_percentage_error
    if name == "accuracy":
        return accuracy_score
    if name == "f1":
        return f1_score
    if name == "log_loss":
        return log_loss

    raise ValueError(f"no support metric {name}")
