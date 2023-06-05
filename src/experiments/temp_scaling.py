from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.CONFIG import DATASETS
from src.helpers.vector_functions import softmax
from src.helpers.load_save_data import load_experiment_data
from src.helpers.logging import setup_logging, get_logger
from src.helpers.train_and_save_calibrators import run_method

setup_logging("temp_scaling")
logger = get_logger("temp_scaling")


########################################################################################################################
#
#  The code for temperature scaling is from https://github.com/dirichletcal/experiments_dnn
#    Meelis Kull, Miquel Perelló-Nieto, Markus Kängsepp, Telmo de Menezes e
#    Silva Filho, Hao Song, and Peter A. Flach. Beyond temperature scaling: Obtaining
#    well-calibrated multiclass probabilities with dirichlet calibration. In NeurIPS, 2019.
#
########################################################################################################################


class TemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs, labels=range(scaled_probs.shape[1]))
        return loss

    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        true = true.argmax(axis=1)
        opt = minimize(self._loss_fun, x0=1, args=(logits, true), options={'maxiter': self.maxiter}, method=self.solver)
        self.temp = opt.x[0]
        logger.info(f"TS temperature param: {self.temp}")

        return opt

    def predict(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)


########################################################################################################################

if __name__ == "__main__":
    for dataset in DATASETS:
        logger.info(f"Running results for dataset: {dataset}")

        run_method(exp_name="TS",
                   exp_data=load_experiment_data(dataset, "uncalibrated_logits"),
                   calibrator_class=TemperatureScaling)
