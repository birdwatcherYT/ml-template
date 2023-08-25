import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    PReLU,
)
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

from .preprocess import Preprocessor
from .tool import get_logger
from .trainer import CVModel, get_kfold, get_metric_func

logger = get_logger(__name__)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class NNModel(tf.keras.Model):
    def __init__(self, cfg, train: pd.DataFrame):
        super().__init__()
        self.cfg: dict = cfg.copy()
        self.cfg_nn: dict = self.cfg["nn"]
        continuous_feat = [
            c for c in self.cfg_nn["feat_col"] if c not in self.cfg["cat_feat"]
        ]
        cat_feat = [c for c in self.cfg_nn["feat_col"] if c in self.cfg["cat_feat"]]
        dropout = self.cfg_nn["dropout"]

        self.cat_embedding = {
            c: Sequential(
                [
                    Input(shape=(1,)),
                    Embedding(train[c].unique().shape[0], 16),
                    Flatten(),
                    *self.dense_activate(8),
                    *self.dense_activate(4),
                ],
                name=f"{c}_emb",
            )
            for c in cat_feat
        }
        self.feature_mlp = Sequential(
            [
                Input(shape=(len(continuous_feat),)),
                BatchNormalization(),
                *self.dense_activate(64),
                BatchNormalization(),
                Dropout(dropout),
                *self.dense_activate(32),
            ]
        )
        self.mlp = Sequential(
            [
                BatchNormalization(),
                *self.dense_activate(128),
                BatchNormalization(),
                Dropout(dropout),
                *self.dense_activate(64),
                BatchNormalization(),
                Dropout(dropout),
                *self.dense_activate(32),
                *self.dense_activate(16),
                *self.dense_activate(16),
                Dense(1, activation=self.cfg_nn["last_activation"]),
            ],
            name="mlp",
        )

    def dense_activate(self, num: int):
        if self.cfg_nn["activation"] == "PReLU":
            return [Dense(num), PReLU()]
        return [Dense(num, activation=self.cfg_nn["activation"])]

    def call(self, inputs: np.ndarray):
        cat_embed = [
            self.cat_embedding[key](inputs[:, i])
            for i, key in enumerate(self.cat_embedding.keys())
        ]
        x = self.feature_mlp(inputs[:, len(self.cat_embedding) :])
        x = Concatenate()([*cat_embed, x])
        x = self.mlp(x)
        return x


class CV_NN(CVModel):
    def __init__(self, cfg):
        self.cfg: dict = cfg
        self.cfg_nn = cfg["nn"]
        self.scaler: StandardScaler = None

        self.models: list[tf.keras.Model] = None
        self.scores: list[float] = None
        self.oof_pred: np.ndarray = None
        self.preps: list[Preprocessor] = None

    def get_feature(self, df: pd.DataFrame, prep: Preprocessor) -> np.ndarray:
        _df = df.copy()
        if prep is not None:
            prep.target_encode(_df)
        # 連続変数
        X = self.scaler.transform(
            df[[c for c in self.cfg_nn["feat_col"] if c not in self.cfg["cat_feat"]]]
        )
        X = np.nan_to_num(X, nan=0)
        # カテゴリ変数
        X = np.concatenate(
            [
                df[
                    [
                        f"{c}_le"
                        for c in self.cfg_nn["feat_col"]
                        if c in self.cfg["cat_feat"]
                    ]
                ],
                X,
            ],
            axis=1,
        )
        return X

    def fit(self, train: pd.DataFrame, seed: int = 1):
        self.scaler = StandardScaler()
        self.scaler.fit(
            train[[c for c in self.cfg_nn["feat_col"] if c not in self.cfg["cat_feat"]]]
        )

        self.oof_pred = np.zeros(train.shape[0])
        self.models = []
        self.scores = []
        self.preps = []

        metric = get_metric_func(self.cfg["metric"])
        for i, (train_index, valid_index) in enumerate(
            get_kfold(self.cfg, train, seed)
        ):
            _train = train.copy()
            prep = None
            if self.cfg["correct_cv"]:
                # target encをtrainでoverride
                prep = Preprocessor(self.cfg)
                prep.aggregate(_train.iloc[train_index])
            self.preps.append(prep)

            X = self.get_feature(_train, prep)
            y = _train[self.cfg["target"]].values

            X_train = X[train_index]
            y_train = y[train_index]
            X_valid = X[valid_index]
            y_valid = y[valid_index]
            logger.info(
                f"cv: {i+1}/{self.cfg['n_splits']}, train_size={len(y_train)}, valid_size={len(y_valid)}"
            )

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg_nn["patience"],
                restore_best_weights=True,
            )
            set_seed(1 + i)
            model = NNModel(self.cfg, train)
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Nadam(
                    learning_rate=self.cfg_nn["learning_rate"]
                ),
            )
            model.fit(
                x=X_train,
                y=y_train,
                batch_size=self.cfg_nn["batch_size"],
                validation_data=(X_valid, y_valid),
                epochs=self.cfg_nn["epochs"],
                callbacks=[early_stopping],
            )

            self.oof_pred[valid_index] = model.predict(X_valid).flatten()
            score = metric(y_valid, self.oof_pred[valid_index])
            logger.info(score)
            self.scores.append(score)
            self.models.append(model)
        logger.info(
            f"nn: oof={metric(y, self.oof_pred)}, mean(score)={np.mean(self.scores)}, scores={self.scores}"
        )

    def predict_array(self, test: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                m.predict(self.get_feature(test, prep)).flatten()
                for m, prep in zip(self.models, self.preps)
            ]
        ).T

    def save(self, outdir: str):
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        with open(outpath / "scaler", "wb") as f:
            joblib.dump(self.scaler, f, compress=3)
        with open(outpath / "scores", "wb") as f:
            joblib.dump(self.scores, f, compress=3)
        with open(outpath / "oof_pred", "wb") as f:
            joblib.dump(self.oof_pred, f, compress=3)
        with open(outpath / "preps", "wb") as f:
            joblib.dump(self.preps, f, compress=3)
        for i, model in enumerate(self.models):
            path = outpath / f"nn_model{i}"
            path.mkdir(parents=True, exist_ok=True)
            model.save_weights(path / "model")

    @classmethod
    def load(cls, cfg, train: pd.DataFrame, outdir: str) -> "CV_NN":
        outpath = Path(outdir)

        model = CV_NN(cfg)

        with open(outpath / "scaler", "rb") as f:
            model.scaler = joblib.load(f)
        with open(outpath / "scores", "rb") as f:
            model.scores = joblib.load(f)
        with open(outpath / "oof_pred", "rb") as f:
            model.oof_pred = joblib.load(f)
        with open(outpath / "preps", "rb") as f:
            model.preps = joblib.load(f)

        model.models = []
        for i in range(len(model.scores)):
            m = NNModel(cfg, train)
            m.load_weights(outpath / f"nn_model{i}" / "model").expect_partial()
            model.models.append(m)
        return model
