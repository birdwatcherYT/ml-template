import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from .cb_trainer import CV_CB
from .lgbm_trainer import CV_LGBM
from .nn_trainer import CV_NN
from .preprocess import Preprocessor
from .tool import CoefficientOptimizer, add_file_hander, get_logger
from .trainer import get_metric_func

logger = get_logger(__name__)

pd.set_option("display.max_columns", 100)


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[{self.name}] start")

    def __exit__(self, *_):
        end_time = time.time()
        logger.info(f"[{self.name}] end: {end_time-self.start_time}")


def load_dataset(data_dir: str):
    # data_dir_path = Path(data_dir)
    # train = pd.read_csv(data_dir_path / "train.csv")
    # test = pd.read_csv(data_dir_path / "test.csv")

    # ダミーデータセット##############
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    datasets = load_diabetes()
    df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
    df["target"] = datasets.target
    df["id"] = np.arange(df.shape[0])
    df["cat"] = KMeans(n_clusters=3, random_state=1, n_init="auto").fit_predict(
        df[[f"s{i+1}" for i in range(6)]]
    )
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=1)
    #################################
    return train, test


@hydra.main(version_base=None, config_name="config", config_path="../yamls")
def main(cfg: DictConfig):
    save_path = Path(cfg.save_path)
    add_file_hander(logger, save_path / __name__)

    preprocessor_save_path = save_path / "preprocessor"
    cb_save_path = save_path / "cb_model"
    lgbm_save_path = save_path / "lgbm_model"
    nn_save_path = save_path / "nn_model"
    co_save_path = save_path / "coefficient"

    config = OmegaConf.to_container(cfg, resolve=True)
    config.update(
        {
            "preprocessor_save_path": preprocessor_save_path.as_posix(),
            "cb_save_path": cb_save_path.as_posix(),
            "lgbm_save_path": lgbm_save_path.as_posix(),
            "nn_save_path": nn_save_path.as_posix(),
            "co_save_path": co_save_path.as_posix(),
        }
    )
    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    with Timer("load_data"):
        train, test = load_dataset(cfg.data_dir)

    exp = config["exp"]
    ensemble_num = exp["ensemble_num"]
    target = exp["target"]

    # 前処理
    with Timer("preprocess"):
        preprocessor = Preprocessor(exp)
        train = preprocessor.preprocess(train, True)
        preprocessor.save(preprocessor_save_path)
        test = preprocessor.preprocess(test, False)

    # 学習
    with Timer("train_lgbm"):
        lgbm_models: list[CV_LGBM] = []
        if exp["lgbm"]["run"]:
            for i in range(ensemble_num):
                logger.info(f"ensemble: {i+1}/{ensemble_num}")
                model = CV_LGBM(exp)
                model.fit(train, i + exp["lgbm"]["seed_base"])
                model.save(lgbm_save_path / f"{i}")
                model.save_importance(lgbm_save_path / f"{i}")
                lgbm_models.append(model)

    with Timer("train_cb"):
        cb_models: list[CV_CB] = []
        if exp["cb"]["run"]:
            for i in range(ensemble_num):
                logger.info(f"ensemble: {i+1}/{ensemble_num}")
                model = CV_CB(exp)
                model.fit(train, i + 1 + exp["cb"]["seed_base"])
                model.save(cb_save_path / f"{i}")
                model.save_importance(cb_save_path / f"{i}")
                cb_models.append(model)

    with Timer("train_nn"):
        nn_models: list[CV_NN] = []
        if exp["nn"]["run"]:
            for i in range(ensemble_num):
                logger.info(f"ensemble: {i+1}/{ensemble_num}")
                model = CV_NN(exp)
                model.fit(train, i + 1 + exp["nn"]["seed_base"])
                model.save(nn_save_path / f"{i}")
                nn_models.append(model)

    all_models = lgbm_models + cb_models + nn_models

    oof_pred = np.array([m.oof_pred for m in all_models]).T
    metric_func = get_metric_func(exp["metric"])
    with Timer("coefficient_optimize"):
        co = CoefficientOptimizer(
            metric_func,
            lower=exp["lower"],
        )
        co.fit(oof_pred, train[target])
        co.save(co_save_path)
        logger.info(
            f"co: score={co.score(oof_pred, train[target])}, coeff={co.coefficients}"
        )

    # 予測
    with Timer("predict"):
        preds = np.array([m.predict(test) for m in all_models]).T
        pred_test = co.predict(preds)
        submit = pd.DataFrame({"id": test["id"], "pred": pred_test})
        submit.to_csv(save_path / "submit.csv", header=False, index=False)
        logger.info(submit)

    logger.info("========== summary ==========")
    oof_scores = []
    for m in all_models:
        score = metric_func(train[target], m.oof_pred)
        logger.info(
            f"{m.__class__.__name__}: oof={score}, mean(scores)={np.mean(m.scores)}, scores={m.scores}"
        )
        oof_scores.append(score)
    logger.info(f"mean(oof_scores)={np.mean(oof_scores)}, oof_scores={oof_scores}")
    logger.info(
        f"co: score={co.score(oof_pred, train[target])}, coeff={co.coefficients}"
    )
    logger.info(f"mean: score={metric_func(train[target], oof_pred.mean(axis=1))}")
    logger.info("========== done ==========")


if __name__ == "__main__":
    with Timer("main"):
        main()
