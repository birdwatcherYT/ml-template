import logging
from functools import partial
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize


def add_file_hander(logger: logging.Logger, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def get_logger(name: str, output_dir: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    if output_dir is not None:
        add_file_hander(Path(output_dir) / name)
    return logger


class CoefficientOptimizer:
    def __init__(self, loss_func, lower=None, initial_coef=None):
        self.loss_func = loss_func
        self.initial_coef = initial_coef
        self.status = dict()
        self.lower = lower

    @property
    def coefficients(self):
        assert "x" in self.status
        w = self.status["x"]
        return np.maximum(self.lower, w) if self.lower is not None else w

    def fit(self, X, y):
        if self.initial_coef is None:
            self.initial_coef = np.ones(X.shape[1]) / X.shape[1]
        loss_partial = partial(self._score, X=X, y=y)
        self.status = minimize(
            loss_partial,
            self.initial_coef,
            method="nelder-mead",
        )

    def _score(self, coef, X, y):
        blend = self.predict(X, coef)
        score = self.loss_func(y, blend)
        return score

    def score(self, X, y, coef=None):
        if coef is None:
            coef = self.coefficients
        return self._score(coef, X, y)

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coefficients
        w = np.maximum(self.lower, coef) if self.lower is not None else coef

        blend = np.dot(X, w)
        return blend

    def save(self, outpath: str):
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "wb") as f:
            joblib.dump(self, f, compress=3)

    @classmethod
    def load(cls, outpath: str) -> "CoefficientOptimizer":
        with open(outpath, "rb") as f:
            return joblib.load(f)
