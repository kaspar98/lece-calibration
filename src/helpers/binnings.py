from copy import copy

import numpy as np


class Binning(object):

    def __init__(self, p, y, n_bins):

        sorted_by_p = np.argsort(p)

        self.p = copy(p)[sorted_by_p]
        self.y = copy(y)[sorted_by_p]

        self.binned_p = self.split_into_bins(self.p)
        self.binned_y = self.split_into_bins(self.y)

        self.n_bins = n_bins
        self.bin_borders = self.get_bin_borders()

    def eval_slope_1(self, pred):
        conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i + 1]) for i in
                      range(len(self.bin_borders) - 2)]
        conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))

        functions = []
        for i in range(self.n_bins):

            if len(self.binned_p[i]) == 0:
                f = lambda pred: 0
            else:
                b = np.mean(self.binned_y[i]) - np.mean(self.binned_p[i])
                f = lambda pred, b=b: 1 * pred + b

            functions.append(f)

        return np.piecewise(pred, conditions, functions)

    def eval_flat(self, pred):
        conditions = [(pred >= self.bin_borders[i]) & (pred < self.bin_borders[i + 1]) for i in
                      range(len(self.bin_borders) - 2)]
        conditions.append(((pred >= self.bin_borders[-2]) & (pred <= 1)))

        functions = []
        for i in range(self.n_bins):

            if len(self.binned_p[i]) == 0:
                f = lambda pred: 0
            else:
                y = np.mean(self.binned_y[i])
                f = lambda pred, y=y: y
            functions.append(f)

        return np.piecewise(pred, conditions, functions)

    def get_ece_abs(self):
        ece = 0
        for i in range(len(self.binned_p)):
            if len(self.binned_p[i]) == 0:
                continue

            mean_p = np.mean(self.binned_p[i])
            mean_y = np.mean(self.binned_y[i])

            ece += np.abs(mean_p - mean_y) * len(self.binned_p[i])
        ece = ece / len(self.p)
        return ece

    def split_into_bins(self, items):
        raise NotImplementedError()

    def get_bin_borders(self):
        raise NotImplementedError()


class EqualWidthBinning(Binning):

    def __init__(self, p, y, n_bins):
        self.n_bins = n_bins
        super().__init__(p, y, n_bins)

    def split_into_bins(self, items):
        bins = [[] for _ in range(self.n_bins)]
        bin_width = 1.0 / self.n_bins

        for item_idx, pred in enumerate(self.p):
            bin_idx = int(pred // bin_width)
            if pred == 1.0:
                bin_idx = self.n_bins - 1

            bins[bin_idx].append(items[item_idx])

        for i in range(len(bins)):
            bins[i] = np.asarray(bins[i])
        return np.asarray(bins, dtype=object)

    def get_bin_borders(self):
        return np.arange(0, 1 + 1e-6, 1 / self.n_bins)


class EqualSizeBinning(Binning):

    def __init__(self, p, y, n_bins):
        self.n_bins = n_bins
        super().__init__(p, y, n_bins)

    def split_into_bins(self, items):
        return np.array_split(items, self.n_bins)

    def get_bin_borders(self):
        bin_borders = [0]
        for b in self.binned_p[1:]:
            bin_borders.append(b[0])
        bin_borders.append(1)
        return np.asarray(bin_borders)
