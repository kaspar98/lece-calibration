import numpy as np

from src import CONFIG
from src.CONFIG import DATASETS
from src.helpers.vector_functions import softmax
from src.helpers.load_save_data import load_dataset, ExperimentData, save_experiment_data, get_path_from_project_root
from src.helpers.logging import setup_logging, get_logger

setup_logging("iop")
logger = get_logger("iop")


def save_iop_experiment_data(dataset: str):
    _, _, _, logits_test, _, y_test = load_dataset(dataset)

    directory_path = get_path_from_project_root() / CONFIG.PRECOMPUTED_RESULTS_DIR / "iop_diag" / dataset
    cal_logits_test = np.load(directory_path / "scores.npy")
    iop_logits_test = np.load(directory_path / "logits.npy")

    # To ensure correct order between IOP experiment and our experiment
    assert (logits_test == iop_logits_test).all()

    cal_p_test = softmax(cal_logits_test)

    experiment_data = ExperimentData(experiment_name="iop_diag",
                                     dataset_name=dataset,
                                     cal_p_test=cal_p_test,
                                     y_test=y_test
                                     )

    save_experiment_data(experiment_data)


if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")

        save_iop_experiment_data(dataset)
