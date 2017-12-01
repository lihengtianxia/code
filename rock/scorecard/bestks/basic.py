# coding=utf-8
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import sys


class _Basic(object):
    @staticmethod
    def basic_prepare(series_, good, bad):
        series_ = series_.replace(
            good, "Good_count").replace(bad, "Bad_count")
        df_ = series_.to_frame()
        df_["_tmp_count_"] = np.ones(df_.shape[0])  # 给定一个具体的个数的1，可以指定数据类型
        return df_

    @staticmethod
    def get_pivot_table(df_n, response, column):
        # print df_n
        df_ = pd.pivot_table(
            df_n, values="_tmp_count_", index=column,
            columns=response, aggfunc=len).fillna(0)
        df_.index.name = "var"
        return df_

    @staticmethod
    def add_basic_info_to_df(df_):
        """添加series
        default_percent:违约率,bad占比
        inside_good_percent:good占所有good的占比
        inside_bad_percent:bad 占所有bad的占比
        :param df_: DataFrame
        :return: DataFrame
        """
        df_["total"] = df_["Good_count"].add(df_["Bad_count"])
        df_["total_percent"] = df_["total"].div(df_["total"].sum())
        df_["default_percent"] = df_["Bad_count"].div(df_["total"])
        df_["inside_good_percent"] = df_["Good_count"].div(
            np.dot(df_["Good_count"].sum(), np.ones(df_.shape[0])))
        df_["inside_bad_percent"] = df_["Bad_count"].div(
            np.dot(df_["Bad_count"].sum(), np.ones(df_.shape[0])))
        return df_

    @staticmethod
    def add_woe_iv_to_df(df_):
        """添加series
        WOE:log(inside_good_percent/inside_bad_percent)
        IV:(inside_good_percent-inside_bad_percent)*WOE
        :param df_: DataFrame
        :return: DataFrame
        """
        df_["WOE"] = df_["inside_good_percent"].div(
            df_["inside_bad_percent"]).map(np.log)  # np.log做log计算,对series做对数计算
        df_["IV"] = (df_["inside_good_percent"].sub(
            df_["inside_bad_percent"])).mul(df_["WOE"])
        df_ = df_.dropna()
        return df_

    @staticmethod
    def add_ks_to_df(df_):
        """添加Series
        cum_good_percent: Good_count累加/Good_count总数
        cum_bad_percent: Bad_count累加/Bad_count总树
        KS: |cum_good_percent-cum_bad_percent|
        :param df_: DataFrame
        :return: DataFrame
        """
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

    @staticmethod
    def check_monotonic(series_, kwargs):
        """检查单调
        """
        check1 = series_.is_monotonic_increasing  # 判断单调递增
        check2 = series_.is_monotonic_decreasing  # 判断单调递减
        if kwargs["strict_monotonicity"]:  # 如果严格单调，返回check1 or check2，都是False才返回False
            return check2 or check1
        else:
            if check2 or check1:
                return True  # 满足严格单调的
            else:
                check = (series_ - series_.shift(1))  # series_.shift(1)移除1个,
                check = check[check != 0].dropna()
                check = check.map(lambda x: 1 if x > 0 else -1)
                check1 = check.is_monotonic_increasing
                check2 = check.is_monotonic_decreasing
                return check2 or check1

    @staticmethod
    def check_proportion(series_, group, kwargs):
        """检查最小占比
        :param series_: 需要切分的series
        :param group: 切分的group
        :param kwargs: 参数
        :return: 验证通过True，否则False
        """
        s_tmp = pd.cut(series_, group, include_lowest=True)
        s_tmp = s_tmp.cat.add_categories([kwargs["fill_value"]]).fillna(kwargs["fill_value"])
        s_tmp = s_tmp.cat.remove_unused_categories()
        df_tmp = pd.DataFrame(s_tmp, columns=["group"])
        counts = df_tmp.groupby(["group"]).size()
        try:
            counts = counts.drop([kwargs["fill_value"]], axis=0)
        except ValueError as e:
            pass
        result = counts.div(series_.shape[0]) > kwargs["min_proportion"]
        if result.all():
            return True
        else:
            return False

    @staticmethod
    def get_tmp_woe(df_n, series_, group, response, kwargs):  # response "code",也就是标签
        df_n[series_.name] = pd.cut(
            series_, group, include_lowest=True)
        df_n[series_.name] = df_n[  # 返回series按照group之后分类的结果，用fill_value的值来填充空的
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


class Progressive(object):
    def __init__(self, total, step=1):
        self.total = total  # 总数据量
        self.percent1 = self.total / 100.0  # 每1%的数据量
        self.now = 0  # 当前进度
        self.step = step  # 显示的步长
        self.prcent_total = int(100 / self.step)

    def bar(self, num, msg=''):
        num += 1
        num = int(num / self.percent1) if num != self.total else 100
        prcent_now = int(num / self.step)
        if num == 100:
            self._print(msg, "=" * prcent_now, " " * (self.prcent_total - prcent_now), num)
            sys.stdout.write('\n')
        elif self.now < prcent_now:
            self.now = prcent_now
            self._print(msg, "=" * prcent_now, " " * (self.prcent_total - prcent_now), num)

    @staticmethod
    def _print(a, b, c, d):
        r = '\r{0}:[{1}{2}]{3}%'.format(a, b, c, d)
        sys.stdout.write(r)
        sys.stdout.flush()
