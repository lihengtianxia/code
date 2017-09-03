import pandas as pd
import numpy as np


class _Basic(object):

    @staticmethod
    def basic_prepare(series_, good, bad):
        series_ = series_.replace(
            good, "Good_count").replace(bad, "Bad_count")
        df_ = series_.to_frame()
        df_["_tmp_count_"] = np.ones(df_.shape[0])
        return df_

    @staticmethod
    def get_pivot_table(df_n, response, column):
        df_ = pd.pivot_table(
            df_n, values="_tmp_count_", index=column,
            columns=response, aggfunc=len).fillna(0)
        df_.index.name = "var"
        return df_

    @staticmethod
    def add_basic_info_to_df(df_):
        df_["total"] = df_["Good_count"].add(df_["Bad_count"])
        df_["default_percent"] = df_["Bad_count"].div(df_["total"])
        df_["inside_good_percent"] = df_["Good_count"].div(
            np.dot(df_["Good_count"].sum(), np.ones(df_.shape[0])))
        df_["inside_bad_percent"] = df_["Bad_count"].div(
            np.dot(df_["Bad_count"].sum(), np.ones(df_.shape[0])))
        return df_

    @staticmethod
    def add_woe_iv_to_df(df_):
        df_["WOE"] = df_["inside_good_percent"].div(
            df_["inside_bad_percent"]).map(np.log)
        df_["IV"] = (df_["inside_good_percent"].sub(
            df_["inside_bad_percent"])).mul(df_["WOE"])
        df_ = df_.dropna()
        return df_

    @staticmethod
    def add_ks_to_df(df_):
        df_ = df_.sort_values("WOE")
        df_["cum_good_percent"] = df_["Good_count"].cumsum().div(
            np.dot(df_["Good_count"].sum(), np.ones(df_.shape[0])))
        df_["cum_bad_percent"] = df_["Bad_count"].cumsum().div(
            np.dot(df_["Bad_count"].sum(), np.ones(df_.shape[0])))
        df_["KS"] = df_["cum_good_percent"].sub(df_["cum_bad_percent"]).abs()
        return df_

    @staticmethod
    def get_iv_ks_std(df_):
        ks_, std = df_["KS"].max(), df_["total"].std()
        iv_sum = df_[(df_["IV"] != np.inf) & (
            df_["IV"] != -np.inf)]["IV"].sum()
        return ks_, std, iv_sum
