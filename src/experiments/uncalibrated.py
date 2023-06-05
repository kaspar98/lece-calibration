from src.CONFIG import DATASETS
from src.helpers.load_save_data import ExperimentData, load_dataset, save_experiment_data
from src.helpers.logging import setup_logging, get_logger

setup_logging("uncalibrated")
logger = get_logger("uncalibrated")


def save_uncalib_experiment_data(dataset: str):
    _, p_train, y_train, _, p_test, y_test = load_dataset(dataset)

    experiment_data = ExperimentData(experiment_name="uncalibrated",
                                     dataset_name=dataset,
                                     cal_p_test=p_test,
                                     cal_p_train=p_train,
                                     y_train=y_train,
                                     y_test=y_test
                                     )

    save_experiment_data(experiment_data)
    logger.info(f"Saved uncalibrated predictions")


def save_uncalib_logit_experiment_data(dataset: str):
    logits_train, _, y_train, logits_test, _, y_test = load_dataset(dataset)

    experiment_data = ExperimentData(experiment_name="uncalibrated_logits",
                                     dataset_name=dataset,
                                     cal_p_test=logits_test,
                                     cal_p_train=logits_train,
                                     y_train=y_train,
                                     y_test=y_test
                                     )

    save_experiment_data(experiment_data)
    logger.info(f"Saved uncalibrated logits")

if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")
        save_uncalib_experiment_data(dataset)
        save_uncalib_logit_experiment_data(dataset)
