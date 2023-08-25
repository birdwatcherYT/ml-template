import datetime
import subprocess
from pathlib import Path

import invoke

ROOT_DIR = Path(__file__).parent

from src.tool import get_logger

logger = get_logger(__name__)

JST = datetime.timezone(datetime.timedelta(hours=+9), "JST")


@invoke.task
def train(c, exp="base", comment: str = "", args: str = ""):
    date = datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    save_path = ROOT_DIR / "model" / f"{date}_{exp}_{comment.replace(' ','_')}"
    save_path.mkdir(parents=True, exist_ok=True)

    cmd = f"cp -r ./src {save_path}"
    logger.info(cmd)
    subprocess.run(cmd, cwd=ROOT_DIR)

    cmd = f"poetry run python -m src.main save_path={save_path.relative_to(ROOT_DIR).as_posix()} exp={exp} {args}"
    logger.info(cmd)
    subprocess.run(cmd, shell=True, cwd=ROOT_DIR)

    logger.info(f"done: {save_path}")


@invoke.task
def predict_test(c, path: str):
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.metrics import mean_absolute_percentage_error

    from src.cb_trainer import CV_CB
    from src.lgbm_trainer import CV_LGBM
    from src.main import load_dataset
    from src.nn_trainer import CV_NN
    from src.preprocess import Preprocessor
    from src.tool import CoefficientOptimizer
    from src.trainer import get_metric_func

    model_path = Path(path)
    with open(model_path / "config.yaml") as f:
        config = yaml.safe_load(f)
    exp = config["exp"]
    n_splits = exp["n_splits"]
    ensemble_num = exp["ensemble_num"]
    print(f"n_splits: {n_splits}")
    print(f"ensemble_num: {ensemble_num}")

    data_dir = Path(config["data_dir"])
    target = exp["target"]

    train, test = load_dataset(data_dir)

    preprocessor_save_path = Path(config["preprocessor_save_path"])
    lgbm_save_path = Path(config["lgbm_save_path"])
    cb_save_path = Path(config["cb_save_path"])
    nn_save_path = Path(config["nn_save_path"])
    co_save_path = Path(config["co_save_path"])

    preprocessor = Preprocessor.load(preprocessor_save_path)
    train = preprocessor.preprocess(train, False)
    test = preprocessor.preprocess(test, False)

    lgbm_models: list[CV_LGBM] = (
        [CV_LGBM.load(lgbm_save_path / f"{i}") for i in range(ensemble_num)]
        if lgbm_save_path.exists()
        else []
    )
    cb_models: list[CV_CB] = (
        [CV_CB.load(cb_save_path / f"{i}") for i in range(ensemble_num)]
        if cb_save_path.exists()
        else []
    )
    nn_models: list[CV_NN] = (
        [CV_NN.load(exp, train, nn_save_path / f"{i}") for i in range(ensemble_num)]
        if nn_save_path.exists()
        else []
    )
    all_models = lgbm_models + cb_models + nn_models

    co = CoefficientOptimizer.load(co_save_path)
    oof_pred = np.array([m.oof_pred for m in all_models]).T

    metric = get_metric_func(exp["metric"])
    for m in all_models:
        print(
            f"{m.__class__.__name__}: oof={metric(train[target], m.oof_pred)}, mean(scores)={np.mean(m.scores)}, scores={m.scores}"
        )

    print(f"co: score={co.score(oof_pred, train[target])}, coeff={co.coefficients}")
    preds = np.array([m.predict(test) for m in all_models]).T
    pred_test = co.predict(preds)

    submit = pd.DataFrame({"id": test["id"], "pred": pred_test})
    print(submit)
    assert np.allclose(np.loadtxt(model_path / "submit.csv", delimiter=","), submit)
