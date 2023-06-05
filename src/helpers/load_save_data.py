import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from src import CONFIG
from src.helpers.vector_functions import softmax


def get_path_from_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent  # helpers <- src <- root


def load_dataset(dataset: str):
    dataset_path = Path(get_path_from_project_root(), CONFIG.LOGIT_DIR, f"probs_{dataset}_logits.p")
    with open(dataset_path, 'rb') as f:
        (logits_train, y_train), (logits_test, y_test) = pickle.load(f)
    p_train = softmax(logits_train)
    p_test = softmax(logits_test)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    y_test = np.eye(len(logits_train[0]))[y_test_flat]
    y_train = np.eye(len(logits_train[0]))[y_train_flat]

    return logits_train, p_train, y_train, \
           logits_test, p_test, y_test


class ExperimentData:

    def __init__(self, experiment_name: str = None,
                 dataset_name: str = None,
                 runtime_inference_test: float = None,
                 runtime_inference_train: float = None,
                 runtime_fitting: float = None,
                 runtime_cross_validation: float = None,
                 cv_best_hyperparams: Dict = {},
                 cv_all_hyperparams: List[Dict] = None,
                 cv_hyperparam_losses: List[float] = None,
                 cv_n_folds: int = None,
                 cal_p_test=None,
                 cal_p_train=None,
                 y_train=None,
                 y_test=None):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.runtime_inference_test = runtime_inference_test
        self.runtime_inference_train = runtime_inference_train
        self.runtime_fitting = runtime_fitting
        self.runtime_cross_validation = runtime_cross_validation
        self.cv_best_hyperparams = cv_best_hyperparams
        self.cv_all_hyperparams = cv_all_hyperparams
        self.cv_hyperparam_losses = cv_hyperparam_losses
        self.cv_n_folds = cv_n_folds
        self.cal_p_test = cal_p_test
        self.cal_p_train = cal_p_train
        self.y_train = y_train
        self.y_test = y_test


def load_experiment_data(dataset: str, experiment_name: str) -> ExperimentData:
    experiment_path = Path(get_path_from_project_root(), CONFIG.RESULTS_DIR, dataset, experiment_name + ".pkl")

    with open(experiment_path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_if_experiment_exists(experiment_name, dataset):
    experiment_path = Path(get_path_from_project_root(), CONFIG.RESULTS_DIR, dataset, experiment_name + ".pkl")
    return experiment_path.exists()


def save_experiment_data(experiment_data: ExperimentData) -> None:
    dataset_results_dir = Path(get_path_from_project_root(), CONFIG.RESULTS_DIR, experiment_data.dataset_name)
    dataset_results_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = Path(dataset_results_dir, experiment_data.experiment_name + ".pkl")

    with open(experiment_path, 'wb') as f:
        pickle.dump(experiment_data, f)


def generate_data(dirichlet, n_data, calibration_function, random_seed=None):

    np.random.seed(random_seed)

    p = np.random.dirichlet(dirichlet, n_data)

    c = calibration_function(p)
    c = c / np.sum(c, axis=1).reshape(-1, 1)

    y = np.array([np.random.choice(np.arange(0, len(dirichlet)), p=pred) for pred in c])
    y = np.eye(len(dirichlet))[y]

    return {"p": p, "c": c, "y": y}