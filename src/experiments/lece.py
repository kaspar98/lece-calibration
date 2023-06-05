import numpy as np

from src.CONFIG import cv_all_hyperparams_LECE_LECD, cv_n_folds, cv_loss, DATASETS
from src.helpers.load_save_data import load_experiment_data
from src.helpers.logging import setup_logging, get_logger
from src.helpers.train_and_save_calibrators import run_method
from src.helpers.vector_functions import normalize_by_sum, minkowski_dist, kullback_leibler

setup_logging("lece")
logger = get_logger("lece")


class NeighborhoodCalibrator():

    def __init__(self, weights_fun, threshold, batch_size, use_LECE_assumption, distance_fun, neighborhood_size):
        self.neighborhood_size = neighborhood_size
        self.distance_fun = distance_fun
        self.weights_fun = weights_fun

        self.threshold = threshold
        self.batch_size = batch_size
        self.use_LECE_assumption = use_LECE_assumption

    def fit(self, p_train, y_train):
        self.p_train = p_train
        self.y_train = y_train

    def predict(self, p_test):
        CE_estimates = np.zeros(p_test.shape)

        for start_idx in range(0, len(p_test), self.batch_size):
            end_idx = start_idx + self.batch_size

            batch_distances = self.distance_fun(p_test=p_test[start_idx:end_idx], p_train=self.p_train)
            batch_weights = self.weights_fun(distances=batch_distances, neighborhood_size=self.neighborhood_size)

            mean_y = np.dot(batch_weights, self.y_train)

            if self.use_LECE_assumption:
                mean_p = np.dot(batch_weights, self.p_train)
                batch_CE_estimates = mean_p - mean_y
            else:
                batch_CE_estimates = p_test[start_idx:end_idx] - mean_y
            CE_estimates[start_idx:end_idx] = batch_CE_estimates

        CE_estimates[p_test <= self.threshold] = 0
        CE_estimates[p_test - CE_estimates <= self.threshold] = 0
        cal_p_test = p_test - CE_estimates
        return normalize_by_sum(cal_p_test)


if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")

        run_method(exp_name="LECE_KL",
                   exp_data=load_experiment_data(dataset, "uncalibrated"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": True,
                                           "distance_fun": kullback_leibler}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)

        run_method(exp_name="TS_LECE_KL",
                   exp_data=load_experiment_data(dataset, "TS"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": True,
                                           "distance_fun": kullback_leibler}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)

        run_method(exp_name="LECE_EUC",
                   exp_data=load_experiment_data(dataset, "uncalibrated"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": True,
                                           "distance_fun": minkowski_dist}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)

        run_method(exp_name="TS_LECE_EUC",
                   exp_data=load_experiment_data(dataset, "TS"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": True,
                                           "distance_fun": minkowski_dist}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)

        run_method(exp_name="LECD_KL",
                   exp_data=load_experiment_data(dataset, "uncalibrated"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": False,
                                           "distance_fun": kullback_leibler}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)

        run_method(exp_name="TS_LECD_KL",
                   exp_data=load_experiment_data(dataset, "TS"),
                   calibrator_class=NeighborhoodCalibrator,
                   cv_all_hyperparams=[{**params,
                                        **{"use_LECE_assumption": False,
                                           "distance_fun": kullback_leibler}}
                                       for params in cv_all_hyperparams_LECE_LECD],
                   cv_n_folds=cv_n_folds,
                   cv_loss=cv_loss)
