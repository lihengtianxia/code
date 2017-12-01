# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
from basic import _Basic
from cut_method_v2 import CutMethods
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
                 group_min_percent=0.05,
                 add_min_group=True,
                 fill_value=-999,
                 strict_monotonicity=True,
                 good=0,
                 bad=1,
                 max_cut_part=10,
                 min_cut_part=2,
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
        if keep_separate_value:  # 如果不为空，则必须是list类型
            if not isinstance(keep_separate_value, list):
                raise ValueError('keep_separate_value must be a list')
        self.keep_separate_value = keep_separate_value
        self.cut_method = None
        self.kwargs = {
            'min_cut_part': min_cut_part,
            'min_proportion': group_min_percent,
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
        part_, single_series, special = CutMethods.cut_method_flow(single_series, **self.kwargs)
        uncheck_len = 2 if self.strict_monotonicity else 3  # 严格单调性
        for group in part_:
            if len(group) < 2:  # 用于special存在的情况
                array.append(group)
            elif len(group) <= uncheck_len:
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
        cut_points = array[-1]
        if special[0]:  # 将特殊点加进去
            if special[1] == 0.01:
                cut_points[0] = special[0] + special[1]
            cut_points.append(special[0])
            cut_points.sort()
        out = pd.cut(single_series, cut_points, include_lowest=True)  # 只留下符合单调的切分后重新切分得到结果
        out = out.cat.add_categories([self.fill_value]).fillna(self.fill_value)
        out = out.cat.remove_unused_categories()
        df_input[single_series.name] = out
        df_input = _Basic.get_pivot_table(df_input, y.name, single_series.name)
        df_input = _Basic.add_basic_info_to_df(df_input)
        df_input = _Basic.add_woe_iv_to_df(df_input)
        # 上面都是重复之前的操作
        df_output = _Basic.add_ks_to_df(df_input)
        df_output.sort_index(inplace=True)
        return cut_points, df_output

    def __bestks_bin(self, single_series, y):
        df_input = _Basic.basic_prepare(y, self.good, self.bad)
        self.kwargs['df_'] = pd.DataFrame(y)
        self.kwargs['response'] = y.name
        part_, single_series, special = CutMethods.cut_method_flow(single_series, **self.kwargs)
        uncheck_len = 2 if self.strict_monotonicity else 3  # 严格单调性
        arrays = []
        for group in part_:
            if len(group) < 2:  # 用于special存在的情况
                arrays.append(group)
            elif len(group) <= uncheck_len:
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
            if special[0]:  # 将特殊点加进去
                if special[1] == 0.01:
                    array[0] = special[0] + special[1]
                array.append(special[0])
                array.sort()
            df_input_tmp = copy.deepcopy(df_input)
            out = pd.cut(single_series, array, include_lowest=True)  # 只留下符合单调的切分后重新切分得到结果
            out = out.cat.add_categories([self.fill_value]).fillna(self.fill_value)
            out = out.cat.remove_unused_categories()
            df_input_tmp[single_series.name] = out
            df_input_tmp = _Basic.get_pivot_table(df_input_tmp, y.name, single_series.name)
            df_input_tmp = _Basic.add_basic_info_to_df(df_input_tmp)
            df_input_tmp = _Basic.add_woe_iv_to_df(df_input_tmp)
            # 上面都是重复之前的操作
            df_output = _Basic.add_ks_to_df(df_input_tmp)
            iv_sum = df_output["IV"].sum()
            if ivs < iv_sum:
                cut_last = array
                ivs = iv_sum
                df_last = copy.deepcopy(df_output)
        df_last.sort_index(inplace=True)
        return cut_last, df_last


class StrBin(object):
    def __init__(self, single_group_num=12, max_of_length=1000,
                 fill_value="-999", good=0, bad=1,
                 group_min_percent=0.05, choice_by='iv',
                 max_cut_part=10, min_cut_part=2,
                 woe_inf_fill="avg"):
        """
        :param single_group_num:int         default:12;当数据总数小于这个数时，直接每个数一个分组
        :param good:
        :param bad:
        :param min_proportion:float         default:0.05;每组的最小占比
        :param max_cut_part:int             default:10;最大分组数,不包含空组
        :param choice_by:str                default:"iv";["iv","ks","woe","len"]其中一个，表示挑选分组时以什么为优先级考虑
        :param woe_inf_fill:str             default:"avg";处理woe中inf,-inf的方法
        """
        self.single_group_num = single_group_num
        self.max_of_length = max_of_length
        self.good = good
        self.bad = bad
        self.fill_value = fill_value
        self.group_min_percent = group_min_percent
        self.max_cut_part = max_cut_part
        self.min_cut_part = min_cut_part
        self.choice_by = choice_by
        self.woe_inf_fill = woe_inf_fill
        self.fill_items = [-np.inf, np.inf, np.nan]

    def single_bin(self, single_series, y):
        """当一列的类别<=12种的时候，直接对这些数据进行分组
        :return: "single",df
        """
        df_input = _Basic.basic_prepare(y, self.good, self.bad)
        df_input[single_series.name] = single_series
        df_input = _Basic.get_pivot_table(df_input, y.name, single_series.name)
        df_input = _Basic.add_basic_info_to_df(df_input)
        df_input = _Basic.add_woe_iv_to_df(df_input)
        df_input = _Basic.add_ks_to_df(df_input)
        df_input = df_input.sort_index()
        return 'single', df_input

    def woe_bin(self, single_series, y):

        df_input = _Basic.basic_prepare(y, bad=self.bad, good=self.good)
        df_input[single_series.name] = single_series
        df_input = _Basic.get_pivot_table(df_input, response=y.name, column=single_series.name)
        df_input = _Basic.add_basic_info_to_df(df_input)
        df_input = _Basic.add_woe_iv_to_df(df_input)
        if self.woe_inf_fill == 'avg':  # todo:填充woe的值，现在是平均值
            avg = df_input["WOE"].replace(self.fill_items, np.nan).mean()
            df_input["WOE"] = df_input["WOE"].replace(self.fill_items, avg)
        df_input = _Basic.add_ks_to_df(df_input)

        last_cut_points = []
        choice_item_value = 0
        parts1 = CutMethods.quantile_cut_flow(df_input['WOE'], max_cut_part=self.max_cut_part, min_cut_part=self.min_cut_part)
        parts2 = CutMethods.cumsum_cut_flow(df_input['WOE'], add_min_group=True, max_cut_part=self.max_cut_part, min_cut_part=self.min_cut_part)
        parts = parts1 + parts2
        for part in parts:  # 目前是取iv最大的组，未来可以加取分组数最多的组
            part = sorted(list(set(part) - {-np.inf, np.inf, np.nan}))
            if len(part) <= 2:
                continue
            bins_ = pd.cut(df_input['WOE'], part, include_lowest=True)
            df_input['bins'] = bins_.cat.codes
            tmp_df = pd.pivot_table(df_input, values=['Bad_count', 'Good_count'], index='bins', aggfunc=np.sum)
            tmp_df = _Basic.add_basic_info_to_df(tmp_df)
            tmp_df['total_percent'] = tmp_df['total'].div(tmp_df['total'].sum())
            if tmp_df['total_percent'].min() < self.group_min_percent:  # 判断每组最小占比
                continue
            tmp_df = _Basic.add_woe_iv_to_df(tmp_df)
            tmp_df = _Basic.add_ks_to_df(tmp_df)
            if self.choice_by.lower() == "iv":
                item_value = tmp_df['IV'].sum()
            elif self.choice_by.lower() == 'woe':
                item_value = tmp_df['WOE'].sum()
            elif self.choice_by.lower() == 'ks':
                item_value = tmp_df['KS'].max()
            elif self.choice_by.lower() == 'len':
                item_value = tmp_df.shape[0]
            else:
                raise ValueError('"choice_by" support ["iv","woe","ks","len"],'
                                 'But get "{0}"'.format(self.choice_by))
            if item_value > choice_item_value:
                last_cut_points = part
        if last_cut_points == []:
            return False, False
        df_input['bins'] = pd.cut(df_input['WOE'], last_cut_points, include_lowest=True).cat.codes
        groups = {}  # 分组号和每组的列表{1:["ZJ","SH"],2:["XJ","BJ"]}
        for i in df_input['bins'].unique():
            items_ = df_input[df_input['bins'] == i].index.tolist()
            groups[i] = items_

        df_ = _Basic.basic_prepare(y, bad=self.bad, good=self.good)
        df_[single_series.name] = single_series

        for code, items in groups.iteritems():
            df_[single_series.name] = df_[single_series.name].replace(items, code)
        df_ = _Basic.get_pivot_table(df_, y.name, single_series.name)
        df_ = _Basic.add_basic_info_to_df(df_)
        df_ = _Basic.add_woe_iv_to_df(df_)
        df_ = _Basic.add_ks_to_df(df_)
        bins_ = pd.Series(['|'.join(i) for i in groups.values()], index=[i for i in groups.keys()], name='var_scope')
        df_ = pd.concat([df_, bins_], axis=1)
        return groups, df_  # ({1:["ZJ","SH"],2:["XJ","BJ"]}, df)

    def fit(self, X, y, columns=None, lable=None, n_jobs=4):
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

        if isinstance(columns, (list, np.ndarray, set)) and isinstance(X, pd.DataFrame):
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
        return self.__get_function(x)(x.fillna(self.fill_value), y)

    def __get_function(self, x):
        """
        get bin function
        :return:
        """
        len_ = len(x.unique())
        if len_ <= 12:
            return self.single_bin
        elif len_ > self.max_of_length:  # 当类型总数大于max_of_length直接返回(False, False)
            f = lambda x, y: (False, False)
            return f
        else:
            return self.woe_bin


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
    for i in range(1000):
        _df[str(i)] = _df.test
    binning = Binning()
    b = binning.fit(_df, lable.code, columns=list(_df.columns))
    print(b)
    # import random
    #
    # s = pd.Series([str(random.randint(0, 22)) for i in range(1000)], name='test')
    # s[100:200] = np.nan
    # y = pd.Series([random.choice([1, 0, 0, 0, 0]) for i in range(1000)], name='code')
    # df = s.to_frame()
    # df['test2'] = s
    # df['test3'] = s
    # sbin = StrBin(choice_by='iv')
    # r = sbin.fit(df, y, columns=['test', 'test2', 'test3'])
    # print(r)
