from sklearn.metrics import log_loss

from src.helpers.vector_functions import uniform_weights_for_proportionally_k_closest

LOGIT_DIR = "logits"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
PRECOMPUTED_RESULTS_DIR = RESULTS_DIR + "/precomputed"

DATASETS = [
    "densenet40_c10",
    "densenet40_c100",
    "resnet_wide32_c10",
    "resnet_wide32_c100",
    "resnet110_c10",
    "resnet110_c100",
]

batch_size = 512
cv_loss = log_loss
cv_n_folds = 10
neighborhood_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.2, 1.0]
thresholds = [0, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.05, 0.10, 1.0]
cv_all_hyperparams_LECE_LECD = [{"neighborhood_size": neighborhood_size,
                                 "threshold": threshold,
                                 "batch_size": batch_size,
                                 "weights_fun": uniform_weights_for_proportionally_k_closest}
                                for neighborhood_size in neighborhood_sizes
                                for threshold in thresholds]
