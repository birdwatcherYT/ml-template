from pathlib import Path

import catboost
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocess import Preprocessor
from .tool import get_logger
from .trainer import CVModel, get_kfold, get_metric_func

logger = get_logger(__name__)


class CV_CB(CVModel):
    def __init__(self, cfg):
        self.cfg: dict = cfg
        self.cfg_cb: dict = cfg["cb"]
        self.models: list[catboost.CatBoost] = None
        self.scores: list[float] = None
        self.oof_pred: np.ndarray = None
        self.preps: list[Preprocessor] = None

    def get_feature(self, df: pd.DataFrame, prep: Preprocessor) -> pd.DataFrame:
        _df = df.copy()
        if prep is not None:
            prep.target_encode(_df)
        _df[self.cfg["cat_feat"]] = (
            _df[self.cfg["cat_feat"]].fillna("nan").astype("category")
        )
        X = _df[self.cfg_cb["feat_col"]]
        return X

    def fit(self, train: pd.DataFrame, seed: int = 1):
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

            X_train = X.iloc[train_index]
            y_train = y[train_index]
            X_valid = X.iloc[valid_index]
            y_valid = y[valid_index]
            logger.info(
                f"cv: {i+1}/{self.cfg['n_splits']}, train_size={len(y_train)}, valid_size={len(y_valid)}"
            )
            cat_index = [
                i
                for i, c in enumerate(self.cfg_cb["feat_col"])
                if c in self.cfg["cat_feat"]
            ]
            dcat_train = catboost.Pool(
                data=X_train,
                label=y_train,
                cat_features=cat_index,
                weight=None,
            )
            dcat_valid = catboost.Pool(
                data=X_valid,
                label=y_valid,
                cat_features=cat_index,
                weight=None,
            )
            model = catboost.CatBoostRegressor(
                boosting_type=self.cfg_cb["boosting_type"],
                iterations=self.cfg_cb["iterations"],
                use_best_model=True,
                eval_metric=self.cfg_cb["eval_metric"],
                depth=self.cfg_cb["depth"],
                loss_function=self.cfg_cb["loss_function"],
                # leaf_estimation_method="Gradient",
                l2_leaf_reg=self.cfg_cb["l2_leaf_reg"],
                learning_rate=self.cfg_cb["learning_rate"],
                random_seed=1 + i,
            )
            model.fit(
                dcat_train,
                eval_set=dcat_valid,
                early_stopping_rounds=self.cfg_cb["early_stopping"],
                verbose=50,
            )
            logger.info(model.best_score_)

            self.oof_pred[valid_index] = model.predict(X_valid)
            score = metric(y_valid, self.oof_pred[valid_index])
            logger.info(score)
            self.scores.append(score)
            self.models.append(model)
        logger.info(
            f"cb: oof={metric(y, self.oof_pred)}, mean(score)={np.mean(self.scores)}, scores={self.scores}"
        )

    def predict_array(self, test: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                m.predict(self.get_feature(test, prep))
                for m, prep in zip(self.models, self.preps)
            ]
        ).T

    def save(self, outdir: str):
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        with open(outpath / "dump", "wb") as f:
            joblib.dump(self, f, compress=3)
        # 念のため個別にも保存
        for i, model in enumerate(self.models):
            model.save_model(outpath / f"cb_model{i}")

    @classmethod
    def load(cls, outdir: str) -> "CV_CB":
        outpath = Path(outdir)
        with open(outpath / "dump", "rb") as f:
            return joblib.load(f)

    def save_importance(self, outdir: str):
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            plt.cla()
            pd.Series(
                model.get_feature_importance(), index=model.feature_names_
            ).sort_values().plot(kind="barh")
            plt.subplots_adjust(left=0.35, right=0.9)
            plt.savefig(outpath / f"cb_importance{i}.png")
            plt.close()
