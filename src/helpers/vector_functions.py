import numpy as np

from src.helpers.binnings import EqualSizeBinning


def confidence_equal_size_ece(p, y, n_bins=15):
    confidence_y = (y.argmax(axis=1) == p.argmax(axis=1)).astype(int)
    confidence_p = p.max(axis=1)
    binning = EqualSizeBinning(p=confidence_p, y=confidence_y, n_bins=n_bins)
    return binning.get_ece_abs()


def classwise_equal_size_ece(p, y, n_bins=15):
    ece_abs_s = []
    for data_class in range(p.shape[1]):
        class_p = p[:, data_class]
        class_y = y[:, data_class]
        binning = EqualSizeBinning(p=class_p, y=class_y, n_bins=n_bins)
        ece_abs_s.append(binning.get_ece_abs())
    return np.mean(ece_abs_s)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


def normalize_by_sum(p):
    return p / np.sum(p, axis=1, keepdims=True)


def bs(p, y):
    return np.mean(np.sum((p - y) ** 2, axis=1))


def minkowski_dist(p_test, p_train, p=2):
    """
    :param p_test: Predictions or logits in shape (rows, features).
    :param p_train: Predictions or logits in shape (rows, features).
    :param p: p=2 for Euclidean, p=1 for Manhattan distance.
    :return: Matrix of distances from every p_test to every p_train instance in shape (nr p_test rows, nr p_train rows)
    """
    return (np.abs(p_test[:, None] - p_train) ** p).sum(axis=2) ** (1 / p)


def kullback_leibler(p_test, p_train):
    """
    :param p_test: Predictions in shape (rows, features).
    :param p_train: Predictions in shape (rows, features).
    :return: Matrix of KL-divergences from every p_test to every p_train instance in shape (nr p_test rows, nr p_train rows)
    """
    p_test_c = np.clip(p_test, a_min=1e-30, a_max=1)
    p_train_c = np.clip(p_train, a_min=1e-30, a_max=1)
    return np.sum(p_test_c[:, None] * np.log(p_test_c[:, None] / p_train_c), axis=2)


def uniform_weights_for_proportionally_k_closest(distances, neighborhood_size=0.1):
    """
    :param distances: Distances from every test instance to every train instance in shape (nr test rows, nr train rows).
    :param neighborhood_size: Neighborhood size as a proportion of the training set size (i.e. k=distances.shape[1] * neighborhood_size).
    :return: For every test instance weights 1/k to k-closest neighbors, weights 0 for others. Output matrix is in input shape.
    """
    k = int(distances.shape[1] * neighborhood_size)

    k_closest_ids = np.argpartition(distances, k - 1)[:, :k]
    weights = np.zeros(distances.shape)
    np.put_along_axis(weights, k_closest_ids, 1 / k, axis=1)
    return weights


def uniform_weights_for_k_closest(distances, neighborhood_size):
    """
    :param distances: Distances from every test instance to every train instance in shape (nr test rows, nr train rows).
    :param neighborhood_size: Number of neighbors (aka 'k').
    :return: For every test instance weights 1/k to k-closest neighbors, weights 0 for others. Output matrix is in input shape.
    """
    k_closest_ids = np.argpartition(distances, neighborhood_size - 1)[:, :neighborhood_size]
    weights = np.zeros(distances.shape)
    np.put_along_axis(weights, k_closest_ids, 1 / neighborhood_size, axis=1)
    return weights