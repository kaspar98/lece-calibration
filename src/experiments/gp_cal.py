from src.helpers.logging import setup_logging, get_logger

setup_logging("GP")
logger = get_logger("GP")

import pycalib.calibration_methods as calm

from src.CONFIG import DATASETS
from src.helpers.load_save_data import load_experiment_data
import numpy as np

from src.helpers.train_and_save_calibrators import run_method


class GPCalibrationCustom:

    def __init__(self, **args):
        self.calibrator = calm.GPCalibration(**args)

    def fit(self, X, y):
        self.calibrator.fit(np.array(X, dtype=float), y.argmax(axis=1))

    def predict(self, X):
        return self.calibrator.predict_proba(np.array(X, dtype=float))


if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")

        exp_data = load_experiment_data(dataset, "uncalibrated_logits")
        n_classes = exp_data.cal_p_train.shape[1]

        run_method(exp_name="GP",
                   exp_data=exp_data,
                   calibrator_class=GPCalibrationCustom,
                   cv_all_hyperparams=[{"n_classes": n_classes, "maxiter": 2000, "n_inducing_points": 10,
                                        "logits": True, "random_state": 0}])
