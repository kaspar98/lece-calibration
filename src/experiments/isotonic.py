import copy

from sklearn.isotonic import IsotonicRegression

from src.CONFIG import DATASETS
from src.helpers.vector_functions import normalize_by_sum
from src.helpers.load_save_data import load_experiment_data
from src.helpers.logging import setup_logging, get_logger
from src.helpers.train_and_save_calibrators import run_method

setup_logging("isotonic")
logger = get_logger("isotonic")


class IsotonicOVR:

    def __init__(self):
        return

    def fit(self, X, y):
        self.n_classes = X.shape[1]
        self.ovr_calibrators = [None] * self.n_classes

        for class_idx in range(self.n_classes):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(X[:, class_idx], y[:, class_idx])
            self.ovr_calibrators[class_idx] = ir

    def predict(self, X):
        cal_p = copy.deepcopy(X)

        for class_idx in range(self.n_classes):
            cal_p[:, class_idx] = self.ovr_calibrators[class_idx].predict(X[:, class_idx]) + 1e-9 * X[:, class_idx]
            # add small amount to avoid infinite log-loss from 0 values

        return normalize_by_sum(cal_p)


if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")

        run_method(exp_name="isotonic",
                   exp_data=load_experiment_data(dataset, "uncalibrated"),
                   calibrator_class=IsotonicOVR)

        run_method(exp_name="TS_isotonic",
                   exp_data=load_experiment_data(dataset, "TS"),
                   calibrator_class=IsotonicOVR)
