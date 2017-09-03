# -*- coding:utf-8 -*-
import logging
import numpy as np
import pandas as pd
from basic import _Basic
from cut_method import _CutMethods
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")


kwargs = {"good": 0, "bad": 1, "strict_monotonicity": True,
          "cut_method": "quantile", "fill_value": "-999",
          "max_cut_part": 10, "add_min_group": True,
          "keep_separate_value": None}


def __get_tmp_woe(df_n, series_, group, response):
    df_n[series_.name] = pd.cut(
        series_, group, include_lowest=True)
    df_n[series_.name] = df_n[
        series_.name].cat.add_categories(
            [kwargs["fill_value"]]).fillna(kwargs["fill_value"])
    df_n[series_.name] = df_n[
        series_.name].cat.remove_unused_categories()
    tmp_table = _Basic.get_pivot_table(
        df_n, response, series_.name)
    tmp_table = _Basic.add_basic_info_to_df(tmp_table)
    tmp_table = _Basic.add_woe_iv_to_df(tmp_table)
    tmp_table = tmp_table.dropna()
    tmp_woe = tmp_table[tmp_table.index !=
                        kwargs["fill_value"]]["WOE"]
    df_n[series_.name] = series_
    return tmp_woe


def __check_monotonic(series_):
    check1 = series_.is_monotonic_increasing
    check2 = series_.is_monotonic_decreasing
    if kwargs["strict_monotonicity"]:
        return check2 or check1
    else:
        if check2 or check1:
            return True
        else:
            check = (series_ - series_.shift(1))
            check = check[check != 0].dropna()
            check = check.map(lambda x: 1 if x > 0 else -1)
            check1 = check.is_monotonic_increasing
            check2 = check.is_monotonic_decreasing
            return check2 or check1


def get_sample():
    df_ = pd.DataFrame()
    df_["code"] = np.zeros(1000)
    df_["code"][:300] = 1
    test = pd.Series(np.random.rand(1000), name="test")
    test[270:350] = np.NaN

    df_input = _Basic.basic_prepare(
        df_["code"], kwargs["good"], kwargs["bad"])
    test = test.round(3)

    array = []
    part_, test = _CutMethods.cut_method_flow(test, kwargs)

    uncheck_len = 4
    if kwargs["strict_monotonicity"]:
        uncheck_len = 3

    for group in part_:
        if len(group) <= uncheck_len:
            array.append(group)
        else:
            try:
                tmp_woe = __get_tmp_woe(df_input, test, group, "code")
                if __check_monotonic(tmp_woe):
                    array.append(group)
                else:
                    break
            except KeyError as error:
                logging.error(error)

    out = pd.cut(test, array[-1], include_lowest=True)
    out = out.cat.add_categories(
        [kwargs["fill_value"]]).fillna(kwargs["fill_value"])
    out = out.cat.remove_unused_categories()
    df_input["test"] = out
    df_input = _Basic.get_pivot_table(df_input, "code", "test")
    df_input = _Basic.add_basic_info_to_df(df_input)
    df_input = _Basic.add_woe_iv_to_df(df_input)
    df_input = _Basic.add_ks_to_df(df_input)
    print df_input

get_sample()
# 可直接运行
