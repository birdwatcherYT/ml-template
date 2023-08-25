from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocess import Preprocessor
from .tool import get_logger
from .trainer import CVModel, get_kfold, get_metric_func

logger = get_logger(__name__)


class CV_LGBM(CVModel):
    def __init__(self, cfg):
        self.cfg: dict = cfg
        self.cfg_lgbm: dict = cfg["lgbm"]
        self.models: list[lgb.Booster] = None
        self.scores: list[float] = None
        self.oof_pred: np.ndarray = None
        self.preps: list[Preprocessor] = None

    def get_feature(self, df: pd.DataFrame, prep: Preprocessor) -> pd.DataFrame:
        _df = df.copy()
        if prep is not None:
            prep.target_encode(_df)
        _df[self.cfg["cat_feat"]] = _df[self.cfg["cat_feat"]].astype("category")
        X = _df[self.cfg_lgbm["feat_col"]]
        return X

    def fit(self, train: pd.DataFrame, seed: int = 1):
        self.models = []
        self.scores = []
        self.oof_pred = np.zeros(train.shape[0])
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

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

            lgbm_params = {
                "objective": self.cfg_lgbm["objective"],
                # "num_class": None,
                "metric": self.cfg_lgbm["metric"],
                "num_leaves": self.cfg_lgbm["num_leaves"],
                "bagging_fraction": self.cfg_lgbm["bagging_fraction"],
                "bagging_freq": self.cfg_lgbm["bagging_freq"],
                "feature_fraction": self.cfg_lgbm["feature_fraction"],
                "learning_rate": self.cfg_lgbm["learning_rate"],
                # "max_depth": 10,
                # "extra_trees": True,
                "seed": 1 + i,
            }
            model = lgb.train(
                lgbm_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                num_boost_round=self.cfg_lgbm["num_boost_round"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.cfg_lgbm["early_stopping"]),
                    lgb.log_evaluation(50),
                ],
            )
            logger.info(model.best_score)

            self.oof_pred[valid_index] = model.predict(
                X_valid, num_iteration=model.best_iteration
            )
            score = metric(y_valid, self.oof_pred[valid_index])
            logger.info(score)
            self.scores.append(score)
            self.models.append(model)
        logger.info(
            f"lgb: oof={metric(y, self.oof_pred)}, mean(score)={np.mean(self.scores)}, scores={self.scores}"
        )

    def predict_array(self, test: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                m.predict(self.get_feature(test, prep), num_iteration=m.best_iteration)
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
            model.save_model(
                outpath / f"lgb_model{i}", num_iteration=model.best_iteration
            )

    @classmethod
    def load(cls, outdir: str) -> "CV_LGBM":
        outpath = Path(outdir)
        with open(outpath / "dump", "rb") as f:
            return joblib.load(f)

    def save_importance(self, outdir: str):
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            for t in ["gain", "split"]:
                plt.cla()
                lgb.plot_importance(
                    model, ignore_zero=False, height=0.5, importance_type=t
                )
                plt.subplots_adjust(left=0.35, right=0.9)
                plt.savefig(outpath / f"lgb_importance{i}_{t}.png")
                plt.close()
