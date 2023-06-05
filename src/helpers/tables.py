import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.helpers.vector_functions import classwise_equal_size_ece, confidence_equal_size_ece, bs
from src.helpers.load_save_data import load_experiment_data


def rename_df(df):
    return df.rename(
        index={"bs": 'Brier score',
               "ll": "log-loss",
               "cw_ece": "classwise ECE",
               "conf_ece": "confidence ECE",
               "resnet_wide32": "ResNet Wide 32",
               "densenet40": "DenseNet-40",
               "resnet110": "ResNet-110",
               "c10": "C-10",
               "c100": "C-100"},
        columns={"LECE_KL": "LECE",
                 "LECE_EUC": "LECE$_{euc}$",
                 "LECD_KL": "LECD",
                 "TS_LECE_KL": "TS+LECE",
                 "TS_LECE_EUC": "TS+LECE$_{euc}$",
                 "TS_LECD_KL": "TS+LECD",
                 "uncalibrated": "uncal",
                 "iop_diag": "IOP",
                 "dec2TS": "TS+DEC",
                 "isotonic": "IR",
                 "TS_isotonic": "TS+IR",
                 })


def add_double_column_header(latex, header_names, header_widths):
    separator_ids = np.cumsum([8, 2])[:-1]
    lines = latex.splitlines()
    output = ""
    for i in range(len(lines)):
        if i != 2:
            output += lines[i] + "\n"
        else:
            for header_idx in range(len(header_names)):
                if header_idx != 0:
                    output += "&"
                output += "\multicolumn{" + str(header_widths[header_idx]) + "}"
                if header_idx != 0:
                    output += "{c}{"
                else:
                    output += "{c}{"
                output += header_names[header_idx] + "}"

            output += "\\\\\n"
            col_names = lines[2].replace(" ", "").replace("\\", "").split("&")
            for col_idx in range(len(col_names)):
                if col_idx != 0:
                    output += "&"
                output += "\multicolumn{1}{"
                if col_idx in separator_ids:
                    output += ""
                output += "c}{" + col_names[col_idx] + "}"
            output += "\\\\"

    return output


def df_to_latex(df):
    with pd.option_context("max_colwidth", 25):
        return df.to_latex(escape=False)


def add_hline(latex, line_ids, lengths):
    lines = latex.splitlines()
    output = ""
    id_idx = 0
    for i in range(len(lines)):
        output += lines[i] + "\n"
        if i in line_ids:
            output += "\cmidrule{" + str(lengths[id_idx][0]) + "-" + str(lengths[id_idx][1]) + "}\n"
            id_idx += 1
    return output


def create_table(datasets, metric, experiment_names, add_avg_rank=True, rounding=3, n_bins=15):
    df = {}

    for dataset_model in datasets:
        model = "_".join(dataset_model.split("_")[:-1])
        dataset = dataset_model.split("_")[-1]

        df[(dataset, model)] = {}

        for experiment_name in experiment_names:
            exp_data = load_experiment_data(dataset=dataset_model, experiment_name=experiment_name)

            if metric == "bs":
                score = bs(exp_data.cal_p_test, exp_data.y_test)
            elif metric == "ll":
                score = log_loss(exp_data.y_test, exp_data.cal_p_test)
            elif metric == "cw_ece":
                score = 100 * classwise_equal_size_ece(p=exp_data.cal_p_test, y=exp_data.y_test, n_bins=n_bins)
            elif metric == "conf_ece":
                score = 100 * confidence_equal_size_ece(p=exp_data.cal_p_test, y=exp_data.y_test, n_bins=n_bins)
            elif metric == "accuracy":
                score = np.mean(exp_data.y_test.argmax(axis=1) == exp_data.cal_p_test.argmax(axis=1))

            df[(dataset, model)][experiment_name] = np.round(score, rounding)

    df = pd.DataFrame.from_dict(df).T
    if metric == "accuracy":
        dfn = add_ranks_to_df(df, ascending=False)
    else:
        dfn = add_ranks_to_df(df, ascending=True)
    if add_avg_rank:
        if metric == "accuracy":
            ranks = df.rank(axis=1, ascending=False, method="min")
        else:
            ranks = df.rank(axis=1, method="min")

        dfn.loc[("", "average rank"), :] = np.round(np.mean(ranks), 1)
    return dfn


def add_ranks_to_df(df_in, ascending=True):
    df = pd.DataFrame.copy(df_in, deep=True)
    ranks = df.rank(axis=1, ascending=ascending, method="min")

    for row_idx in range(len(df)):
        for column_idx in range(len(df.iloc[row_idx])):

            item = df.iloc[row_idx, column_idx]
            rank = ranks.iloc[row_idx, column_idx]

            item = np.round(item, 7)
            df.iloc[row_idx, column_idx] = str(item) + "_{" + str(int(rank)) + "}"
            if rank == 1:
                df.iloc[row_idx, column_idx] = "\mathbf{" + df.iloc[row_idx, column_idx] + "}"
            df.iloc[row_idx, column_idx] = "$" + df.iloc[row_idx, column_idx] + "$"
    return df
