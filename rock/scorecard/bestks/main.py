# coding=utf-8
import logging
import numpy as np
import pandas as pd
from basic import _Basic
from cut_method import _CutMethods
import copy

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")

kwargs = {
    "good": 0,
    "bad": 1,
    "strict_monotonicity": True,  # 是否需要严格单调strict_monotonicity
    "cut_method": "quantile",  # cut_method 目前可选"cumsum","kestks"
    "fill_value": "-999",
    "max_cut_part": 10,
    "add_min_group": True,
    "keep_separate_value": None,
    "min_proportion": 0.05,  # 每个分割的区域最小占比
    "loop": 3,  # best_ks循环的次数，3次就是找2**3-1个点，分成2**3个区间
}


def get_sample(df_, test):
    kwargs["cut_method"] = "quantile"
    print "*****************************{0}**************************************".format(kwargs["cut_method"])
    df_input = _Basic.basic_prepare(
        df_["code"], kwargs["good"], kwargs["bad"])
    array = []
    part_, test = _CutMethods.cut_method_flow(test, kwargs)
    # part_各个分位数的值,test原始数据
    uncheck_len = 4
    if kwargs["strict_monotonicity"]:  # 严格单调性
        uncheck_len = 3

    for group in part_:
        if len(group) <= uncheck_len:
            if _Basic.check_proportion(test, group, kwargs):
                array.append(group)
        else:
            try:  # 只检测了单调性，可以增加检测分组的数量,增加个参数每个组至少0.05
                if _Basic.check_proportion(test, group, kwargs):
                    tmp_woe = _Basic.get_tmp_woe(df_input, test, group, "code", kwargs)
                    if _Basic.check_monotonic(tmp_woe, kwargs):
                        array.append(group)
                        #     else:
                        #         break
                        # else:  # 不满足单调舍弃
                        #     break
            except KeyError as error:
                logging.error(error)
    print(array)
    out = pd.cut(test, array[-1], include_lowest=True)  # 只留下符合单调的切分后重新切分得到结果
    out = out.cat.add_categories(
        [kwargs["fill_value"]]).fillna(kwargs["fill_value"])
    out = out.cat.remove_unused_categories()
    df_input["test"] = out
    df_input = _Basic.get_pivot_table(df_input, "code", "test")
    df_input = _Basic.add_basic_info_to_df(df_input)
    df_input = _Basic.add_woe_iv_to_df(df_input)
    # 上面都是重复之前的操作
    df_output = _Basic.add_ks_to_df(df_input)
    print df_output
    print "*****************************{0}**************************************".format(kwargs["cut_method"])


def get_sample_bestks(df_, test):
    kwargs["cut_method"] = "bestks"
    print "*****************************{0}**************************************".format(kwargs["cut_method"])
    part_, test = _CutMethods.cut_method_flow(test, kwargs, response="code", df_=df_)
    # groups为切割的组合，后面进行排列组合计算iv的值
    df_input = _Basic.basic_prepare(df_["code"], kwargs["good"], kwargs["bad"])
    uncheck_len = 4
    if kwargs["strict_monotonicity"]:  # 严格单调性
        uncheck_len = 3
    arrays = []
    for group in part_:
        if len(group) <= uncheck_len:
            if _Basic.check_proportion(test, group, kwargs):
                arrays.append(group)
        else:
            try:  # 只检测了单调性，可以增加检测分组的数量,增加个参数每个组至少0.05
                if _Basic.check_proportion(test, group, kwargs):
                    tmp_woe = _Basic.get_tmp_woe(df_input, test, group, "code", kwargs)
                    if _Basic.check_monotonic(tmp_woe, kwargs):
                        arrays.append(group)
            except KeyError as error:
                logging.error(error)
    # 筛选出符合单调的要求和分布要求的,再筛选出iv最大的一个
    ivs = 0
    df_last = None
    for array in arrays:
        df_input_tmp = copy.deepcopy(df_input)
        out = pd.cut(test, array, include_lowest=True)  # 只留下符合单调的切分后重新切分得到结果
        out = out.cat.add_categories([kwargs["fill_value"]]).fillna(kwargs["fill_value"])
        out = out.cat.remove_unused_categories()
        df_input_tmp["test"] = out
        df_input_tmp = _Basic.get_pivot_table(df_input_tmp, "code", "test")
        df_input_tmp = _Basic.add_basic_info_to_df(df_input_tmp)
        df_input_tmp = _Basic.add_woe_iv_to_df(df_input_tmp)
        # 上面都是重复之前的操作
        df_output = _Basic.add_ks_to_df(df_input_tmp)
        iv_sum = df_output["IV"].sum()
        # print df_output
        if ivs < iv_sum:
            ivs = iv_sum
            df_last = copy.deepcopy(df_output)
    print df_last
    print "*****************************{0}**************************************".format(kwargs["cut_method"])


if __name__ == '__main__':
    df_ = pd.DataFrame()
    df_["code"] = np.zeros(100000)
    df_["code"][30000:60000] = 1
    df_["code"][80000:96000] = 1
    test1 = pd.Series(np.random.randint(10, 100, 50000))
    test2 = pd.Series(np.random.randint(40, 60, 50000))
    test = test1.append(test2, ignore_index=True)  # ignore_index pandas 0.19才有
    test.name = "test"
    test[22000:40000] = np.NaN
    test[70000:90000] = np.NaN
    test = test.round(3)
    # get_sample(df_, test)
    get_sample_bestks(df_, test)

    # import pandas as pd
    #
    # df_ = pd.read_csv("bestks_test.csv")
    # df_.drop("Unnamed: 0", axis=1)
    # df_ = df_.drop("Unnamed: 0", axis=1)
    # df_.columns = ["info1", "info2", "code"]
    # test = df_["info2"]
    # df = pd.DataFrame()
    # df["code"] = df_["code"]
    # test = test.round(3)
    # get_sample(df_, test)
    # get_sample_bestks(df, test)
