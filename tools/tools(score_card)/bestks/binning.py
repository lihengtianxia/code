# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
from bestks.basic import _Basic
from bestks.cut_method import _CutMethods
import pandas as pd
import numpy as np
import logging
import copy
import multiprocessing


# logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class Binning(object):
    general_bin_types = ['quantile', 'cumsum']

    def __init__(self,
                 cut_method='quantile',
                 min_proportion=0.05,
                 add_min_group=True,
                 fill_value=-999,
                 strict_monotonicity=True,
                 good=0,
                 bad=1,
                 max_cut_part=10,
                 keep_separate_value=None,
                 loop=3
                 ):
        """

        :param cut_method: str
            One of [quantile, cumsum,kestks]
        :param min_proportion: float
            0>min_proportion>1.0, every area minimum ratio. default 0.05;
        :param add_min_group: Boolean
            add min group. default True;
        :param fill_value: int
            Fill empty values. default -999;
        :param strict_monotonicity: Boolean
            Do you need strict monotonicity, default True;
        :param good: str or int
            Positive samples lable. default 0;
        :param bad: str or int
            Inactive samples lable. default 1;
        :param loop: int
            just useful for best-ks, search deep;

        """
        self.min_proportion = None
        self.add_min_group = None
        self.fill_value = None
        self.strict_monotonicity = None
        self.good = None
        self.bad = None
        self.loop = None
        self.max_cut_part = None
        self.keep_separate_value = None
        self.cut_method = None
        self.kwargs = {
            'min_proportion': min_proportion,
            'add_min_group': add_min_group,
            'fill_value': fill_value,
            'strict_monotonicity': strict_monotonicity,
            'good': good,
            'bad': bad,
            'loop': loop,
            'max_cut_part': max_cut_part,
            'keep_separate_value': keep_separate_value,
            'cut_method': cut_method
        }
        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y, columns=None, lable=None, n_jobs=4):
        """

        :param X: DataFrame or Series
        :param y: DataFrameor or Series or list
        :param columns: list or str
            bin Series name;
            If X is DataFrame, columns is necessary;
        :param lable: str
            target Series name;
            If y is DataFrame, lable is necessary;
        :param n_jobs: int
            bin thread num
        :return: dict or tuple
            if columns is list: dict
            if columns is str: tuple

        """
        if isinstance(y, list):
            y = pd.Series(y)
            y.name = 'code'
        elif isinstance(y, pd.DataFrame):
            if lable is None:
                raise ValueError('If y is DataFrame, lable  can not None.May str')
            y = y[lable]

        if isinstance(X, pd.DataFrame):
            if columns is None:
                raise ValueError('If X is DataFrame, columns can not None.May str or list')
            single_series = X[columns]
        else:
            single_series = X
        # y pd.Series, single_series = Series
        if y.shape[0] != single_series.shape[0]:
            raise ValueError('X.shape[0] != y.shape[0]')

        if isinstance(columns, list) and isinstance(X, pd.DataFrame):
            result = []
            pool = multiprocessing.Pool(processes=n_jobs)
            for i in columns:
                result.append(pool.apply_async(self, (X[i], y)))
            pool.close()
            pool.join()
            return dict(zip(columns, [res.get() for res in result]))
        else:
            return self.__call__(single_series, y)

    def __call__(self, x, y):
        return self.__get_function()(x, y)

    def __get_function(self):
        """
        get bin function
        :return:
        """
        if self.cut_method in self.general_bin_types:
            return self.__general_bin
        elif self.cut_method == 'bestks':
            return self.__bestks_bin
        else:
            raise ValueError('{0} is not support!'.format(self.cut_method))

    def __general_bin(self, single_series, y):
        array = []
        df_input = _Basic.basic_prepare(y, self.good, self.bad)
        part_, single_series = _CutMethods.cut_method_flow(single_series, self.kwargs)
        uncheck_len = 3 if self.strict_monotonicity else 4  # 严格单调性
        for group in part_:
            if len(group) <= uncheck_len:
                if _Basic.check_proportion(single_series, group, self.kwargs):
                    array.append(group)
            else:
                try:  # 只检测了单调性，可以增加检测分组的数量,增加个参数每个组至少0.05
                    if _Basic.check_proportion(single_series, group, self.kwargs):
                        tmp_woe = _Basic.get_tmp_woe(df_input, single_series, group, y.name, self.kwargs)
                        if _Basic.check_monotonic(tmp_woe, self.kwargs):
                            array.append(group)
                except KeyError as error:
                    logging.error(error)
        if not array:
            return False
        out = pd.cut(single_series, array[-1], include_lowest=True)  # 只留下符合单调的切分后重新切分得到结果
        out = out.cat.add_categories([self.fill_value]).fillna(self.fill_value)
        out = out.cat.remove_unused_categories()
        df_input[single_series.name] = out
        df_input = _Basic.get_pivot_table(df_input, y.name, single_series.name)
        print (df_input)
        df_input = _Basic.add_basic_info_to_df(df_input)
        df_input = _Basic.add_woe_iv_to_df(df_input)
        # 上面都是重复之前的操作
        df_output = _Basic.add_ks_to_df(df_input)
        # print(df_output)
        return array[-1], df_output

    def __bestks_bin(self, single_series, y):
        df_input = _Basic.basic_prepare(y, self.good, self.bad)
        part_, single_series = _CutMethods.cut_method_flow(single_series, self.kwargs, response=y.name, df_=pd.DataFrame(y))
        uncheck_len = 3 if self.strict_monotonicity else 4  # 严格单调性
        arrays = []
        for group in part_:
            if len(group) <= uncheck_len:
                if _Basic.check_proportion(single_series, group, self.kwargs):
                    arrays.append(group)
            else:
                try:  # 只检测了单调性，可以增加检测分组的数量,增加个参数每个组至少0.05
                    if _Basic.check_proportion(single_series, group, self.kwargs):
                        tmp_woe = _Basic.get_tmp_woe(df_input, single_series, group, y.name, self.kwargs)
                        if _Basic.check_monotonic(tmp_woe, self.kwargs):
                            arrays.append(group)
                except KeyError as error:
                    logging.error(error)

        ivs = 0
        df_last = None
        cut_last = []
        if not arrays:
            return False
        for array in arrays:
            df_input_tmp = copy.deepcopy(df_input)
            out = pd.cut(single_series, array, include_lowest=True, right=False)  # 只留下符合单调的切分后重新切分得到结果
            out = out.cat.add_categories([self.fill_value]).fillna(self.fill_value)
            out = out.cat.remove_unused_categories()
            df_input_tmp[single_series.name] = out
            df_input_tmp = _Basic.get_pivot_table(df_input_tmp, y.name, single_series.name)
            df_input_tmp = _Basic.add_basic_info_to_df(df_input_tmp)
            df_input_tmp = _Basic.add_woe_iv_to_df(df_input_tmp)
            # 上面都是重复之前的操作
            df_output = _Basic.add_ks_to_df(df_input_tmp)
            iv_sum = df_output["IV"].sum()
            # print df_output
            if ivs < iv_sum:
                cut_last = array
                ivs = iv_sum
                df_last = copy.deepcopy(df_output)
        # print(df_last)
        return cut_last, df_last


if __name__ == '__main__':
    lable = pd.DataFrame()
    lable["code"] = np.zeros(100000)
    lable["code"][30000:60000] = 1
    lable["code"][80000:96000] = 1
    s1 = pd.Series(np.random.randint(10, 100, 50000))
    s2 = pd.Series(np.random.randint(40, 60, 50000))
    s = s1.append(s2, ignore_index=True)  # ignore_index pandas 0.19才有
    s.name = "test"
    s[22000:40000] = np.NaN
    s[70000:90000] = np.NaN
    s = s.round(3)
    _df = s.to_frame()
    for i in range(1):
        _df[str(i)] = _df.test
    binning = Binning()
    binning.fit(_df, lable.code, columns=list(_df.columns))
