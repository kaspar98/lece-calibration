import copy
import time

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from src.helpers.vector_functions import softmax
from src.helpers.load_save_data import save_experiment_data, ExperimentData, check_if_experiment_exists
from src.helpers.logging import get_logger

logger = get_logger(__name__)


def find_parameter_losses_with_cv(p, y, calibrator_class, cv_params_to_try, n_cv_folds, loss_fun):
    cv_scores = [0] * len(cv_params_to_try)
    for cv_params_idx, cv_params in enumerate(cv_params_to_try):
        logger.info(f"trying params {cv_params_idx}/{len(cv_params_to_try)}")
        logger.info(cv_params)
        fold_scores = []
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=0)

        for train_index, test_index in kf.split(p):
            p_train, p_test = p[train_index], p[test_index]
            y_train, y_test = y[train_index], y[test_index]

            calibrator = calibrator_class(**cv_params)
            calibrator.fit(p_train, y_train)
            cal_p_test = calibrator.predict(p_test)

            loss = loss_fun(y_test, cal_p_test)
            fold_scores.append(loss)

        cv_scores[cv_params_idx] = np.mean(fold_scores)

        logger.info(f"loss: {np.round(np.mean(fold_scores), 6)}")

    return cv_scores


def run_method(exp_name,
               exp_data: ExperimentData,
               calibrator_class,
               cv_n_folds=None,
               cv_loss=log_loss,
               cv_all_hyperparams=None,
               ):
    logger.info(f"running '{exp_name}'")
    if check_if_experiment_exists(exp_name, exp_data.dataset_name):
        logger.info(f"experiment already exists, returning")
        return

    X_train = exp_data.cal_p_train
    X_test = exp_data.cal_p_test
    y_train = exp_data.y_train
    y_test = exp_data.y_test

    if (X_test >= 0).all() and (X_test <= 1).all():
        logger.info(f"loss before: {cv_loss(y_test, X_test)}")
    else:
        logger.info(f"loss before: {cv_loss(y_test, softmax(X_test))}")

    # Hyperparameter tuning
    cv_hyperparam_losses = None
    runtime_cross_validation = None
    if cv_all_hyperparams is None:
        cv_best_hyperparams = {}
    elif len(cv_all_hyperparams) == 1:
        cv_best_hyperparams = cv_all_hyperparams[0]
    else:
        start = time.process_time()
        cv_hyperparam_losses = find_parameter_losses_with_cv(p=X_train, y=y_train,
                                                             calibrator_class=calibrator_class,
                                                             cv_params_to_try=cv_all_hyperparams,
                                                             n_cv_folds=cv_n_folds, loss_fun=cv_loss)
        runtime_cross_validation = time.process_time() - start
        cv_best_hyperparams = cv_all_hyperparams[np.argmin(cv_hyperparam_losses)]

    calibrator = calibrator_class(**cv_best_hyperparams)
    # Fitting
    start = time.process_time()
    calibrator.fit(X_train, y_train)
    runtime_fitting = time.process_time() - start

    # Inference test
    start = time.process_time()
    cal_p_test = calibrator.predict(X_test)
    runtime_inference_test = time.process_time() - start

    # Inference train
    start = time.process_time()
    cal_p_train = calibrator.predict(X_train)
    runtime_inference_train = time.process_time() - start

    # Update experiment info and save
    new_exp_data = copy.deepcopy(exp_data)
    new_exp_data.experiment_name = exp_name

    new_exp_data.runtime_inference_test = runtime_inference_test
    new_exp_data.runtime_inference_train = runtime_inference_train
    new_exp_data.runtime_fitting = runtime_fitting
    new_exp_data.runtime_cross_validation = runtime_cross_validation

    new_exp_data.cv_best_hyperparams = cv_best_hyperparams
    new_exp_data.cv_all_hyperparams = cv_all_hyperparams
    new_exp_data.cv_hyperparam_losses = cv_hyperparam_losses
    new_exp_data.cv_n_folds = cv_n_folds

    new_exp_data.cal_p_test = cal_p_test
    new_exp_data.cal_p_train = cal_p_train

    save_experiment_data(experiment_data=new_exp_data)

    # Log to console
    logger.info(f"finished experiment: {exp_name}")
    logger.info(f"cv_param: {cv_best_hyperparams}")
    logger.info(f"loss after: {cv_loss(y_test, cal_p_test)}")
    logger.info(f"runtime_cross_validation: {runtime_cross_validation}")
    logger.info(f"runtime_inference_test: {runtime_inference_test}")

