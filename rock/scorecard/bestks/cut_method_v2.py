# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import copy
from itertools import combinations


class CutMethods(object):
    @staticmethod
    def cut_method_flow(series, **kwargs):
        """
        max_cut_part,min_cut_part,keep_separate_value,add_min_group
        """
        cut_method = {
            "quantile": CutMethods.quantile_cut_flow,
            "cumsum": CutMethods.cumsum_cut_flow,
            "bestks": CutMethods.best_ks_cut_flow
        }
        keep_separate_value = kwargs.get('keep_separate_value')
        separate, state = None, None
        if keep_separate_value:
            keep_separate_value.sort()
            for value in keep_separate_value:
                separate, state = CutMethods.check_separate(series, value)
                if state:
                    break

        function = cut_method.get(kwargs['cut_method'])

        if separate is None:
            part_ = function(series, **kwargs)
            special = None
        else:
            kwargs_ = copy.deepcopy(kwargs)
            kwargs_['max_cut_part'] = kwargs.get('max_cut_part') - 1
            kwargs_['min_cut_part'] = kwargs.get('min_cut_part') - 1
            kwargs_['separate'] = separate
            # series_ = series.replace({separate: separate + state})
            series_ = series[series != separate]
            part_ = function(series_, **kwargs_)
            special = separate
        return part_, series, (special, state)

    @staticmethod
    def quantile_cut_flow(series, **kwargs):
        max_cut_part = kwargs.get('max_cut_part', 10)
        min_cut_part = kwargs.get('min_cut_part', 2)
        part_all = []
        for i in range(min_cut_part, max_cut_part + 1):
            part = series.quantile(np.linspace(0, 1, i + 1)).unique().tolist()
            part.sort()
            if part not in part_all:
                part_all.append(part)
        return part_all

    @staticmethod
    def cumsum_cut_flow(series, **kwargs):
        def _cumsum_cut(values, series, gap, **kwargs):
            series_min = series.min()
            series_max = series.max()
            add_min_group = kwargs.get('add_min_group', True)
            iterable = values.iteritems()
            cum, cut_list = 0, []
            for value, sum_ in iterable:
                cum += sum_
                if cum >= gap:
                    cut_list.append(value)
                    cum = 0
            if kwargs.get('separate', None) is None:  # 如果不存在指定最小值的情况，add_min_group才会生效
                if (cut_list[0] == series_min) and add_min_group:
                    cut_list.append(series_min + 0.01)
            if series_min not in cut_list:
                cut_list.append(series_min)
            if series_max not in cut_list:
                cut_list.append(series_max)
            return cut_list

        part_all = []
        max_cut_part = kwargs.get('max_cut_part', 10)
        min_cut_part = kwargs.get('min_cut_part', 2)
        values = series.value_counts(normalize=True).sort_index()
        for i in range(min_cut_part, max_cut_part + 1):
            part = _cumsum_cut(values, series, round(1.0 / i, 2), **kwargs)
            part.sort()
            if part not in part_all:
                part_all.append(part)
        return part_all

    @staticmethod
    def check_separate(series, value):
        separate, state = None, None
        if value == series.min():
            state = 0.01
            separate = value
        elif value == series.max():
            state = - 0.01
            separate = value
        else:
            pass
        return separate, state

    @staticmethod
    def best_ks_cut_flow(series_, **kwargs):
        """best_ks切分
        :param df_: 带标签的DataFrame
        :param series_: 数据列Series
        :param response: 标签列的名称
        :param kwargs: 参数
        :return: 分割组合,数据列
        """
        # df_ = kwargs.get('df_')
        # response = kwargs.get('response')
        df_tmp = CutMethods._get_pivot_for_bestks(series_, **kwargs)
        bestks = BestKS(df_tmp, loop=kwargs["loop"])
        part_ = bestks.get_part(loop=kwargs["loop"])
        return part_

    @staticmethod
    def _get_pivot_for_bestks(series_, df_, response, **kwargs):
        """bestks时需要的初始化部分代码
        :param df_: 带标签的DataFrame
        :param series_: 数据列Series
        :param response: 标签列的名称
        :param kwargs: 参数
        :return: DataFrame
        """
        df_ = df_.replace(kwargs["good"], "Good_count").replace(kwargs["bad"], "Bad_count")
        df_["data"] = series_.fillna(kwargs["fill_value"])  # todo:不确定，替换nan为-999
        # 第一步转制,
        df_tmp = pd.pivot_table(df_, index=["data"], columns=[response], aggfunc=len, fill_value=0)
        try:
            df_tmp = df_tmp.drop(kwargs["fill_value"], 0)
        except ValueError as e:
            pass
        return df_tmp


class BestKS(object):
    """
    df_ = pd.DataFrame()
    df_["code"] = np.zeros(1000)
    df_["code"][300:600] = 1
    test1 = pd.Series(np.random.randint(10, 100, 500))
    test2 = pd.Series(np.random.randint(40, 60, 500))
    test = test1.append(test2, ignore_index=True)
    test[220:400] = np.NaN
    df_tmp = get_pivot_first(df_, test, "code", kwargs)
    df_tmp = df_tmp.drop(kwargs["fill_value"], 0)
    # 生成测试数据
    bestks = BestKS(df_tmp, loop=3)
    print bestks.get_points(loop=3)
    """

    def __init__(self, df, loop=3, depth=1, parent=None):
        """depth=1, parent=None初始化类的时候切记不要定义!
        :param df: DataFrame
        :param loop: 循环的次数
        :param depth: 当前节点的深度
        :param parent: 当前节点的父节点
        """
        if depth <= loop:
            self.df = df
            self.max_ks = ""  # 最大ks的value
            self.left_df = ""  # 左半边数据
            self.right_df = ""  # 右半边数据
            self.left_node = None  # 左节点
            self.right_node = None  # 右节点
            self.parent_node = parent  # 父亲节点
            self.loop = loop
            self.depth = depth  # 深度
            self._set_init()

    def _set_init(self):
        self.max_ks = self._get_max_ks_value(self.df)
        self.left_df, self.right_df = self._get_left_right(self.df, self.max_ks)
        if self.right_df.shape[0] < 3:
            self.right_node = None
        else:
            self.right_node = BestKS(self.right_df, depth=self.depth + 1, parent=self)
        if self.left_df.shape[0] < 3:
            self.left_node = None
        else:
            self.left_node = BestKS(self.left_df, depth=self.depth + 1, parent=self)

    @staticmethod
    def _get_left_right(df_, max_ks):
        """根据max_ks得到左右两边的数据继续算ks
        :param df_: DataFrame
        :param max_ks: max_ks的值
        :return: (DataFrame,DataFrame)
        """
        df_left = df_.ix[df_.index < max_ks, ["Bad_count", "Good_count"]]
        df_right = df_.ix[df_.index >= max_ks, ["Bad_count", "Good_count"]]
        return df_left, df_right

    @staticmethod
    def _get_max_ks_value(df_):
        """得到最大ks的value
        :param df_: DataFrame
        :return: value
        """
        # 第二步计算ks
        df_["cum_good_percent"] = df_["Good_count"].cumsum().div(
            np.dot(df_["Good_count"].sum(), np.ones(df_.shape[0])))
        df_["cum_bad_percent"] = df_["Bad_count"].cumsum().div(
            np.dot(df_["Bad_count"].sum(), np.ones(df_.shape[0])))
        df_["KS"] = df_["cum_good_percent"].sub(df_["cum_bad_percent"]).abs()
        max_ks = df_.idxmax()["KS"]  # 得到最大的ks的value
        # max_ks_value = df_tmp.get_value(max_ks, "KS")#得到ks值
        return max_ks

    def _get_points(self, loop, points=None):
        """得到bestks分割的几个点
        :param loop: 层数
        :param points: 不用填写
        :return: points list [1,2,3,4]
        """
        if not points:
            points = []
        points.append(self.max_ks)
        if self.depth < loop:
            if self.left_node:
                self.left_node._get_points(loop=loop - 1, points=points)
            if self.right_node:
                self.right_node._get_points(loop=loop - 1, points=points)
        else:
            if self.left_node:
                points.extend([self.left_node.max_ks])
            if self.right_node:
                points.extend([self.right_node.max_ks])
        return points

    def get_part(self, loop=None):
        """返回所有切割的组合
        :param loop:  层数,默认为self.loop
        :return: group list[list] [[1,2,3,4]]
        """
        max_p = self.df.index.max()
        min_p = self.df.index.min()
        groups = []
        if not loop:
            loop = self.loop
        points = [p for p in self._get_points(loop=loop) if str(p) not in [str(max_p), str(min_p), 'nan']]
        points = list(set(points))
        for i in range(1, len(points)):
            for p in combinations(points, i):
                pl = list(p)
                pl.extend([max_p, min_p])
                pl.sort()
                groups.append(pl)
        return groups


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    s = pd.Series(np.random.randint(10, 200, 100))
    a, b, c = CutMethods.cut_method_flow(s, max_cut_part=10,
                                         min_cut_part=2, keep_separate_value=[10, 12],
                                         cut_method='cumsum')
    print(a)
    print(b.min())
