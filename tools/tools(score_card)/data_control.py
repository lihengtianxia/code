# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import datetime
from bestks.binning import Binning
from bestks.basic import _Basic
from bestks.cut_method import _CutMethods


def log(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


class DcBins(object):
    def __init__(self, df_train, df_test=None,
                 drop_cols=None, bad=1, good=0,
                 replace_options=None, replace_all=None,
                 cut_method="cumsum", max_cut_part=10, min_group_num=3, group_min_percent=0.05,
                 check_monotonicity=True, strict_monotonicity=True,
                 fill_value="-999", response='code',
                 enable_iv_threshold=True, iv_min=0.02, keep_range=500,
                 enable_single_threshold=True, single_threshold=0.8, include_none=True,
                 n_jobs=4, woe_fill='avg'):
        """
        堡垒机环境下使用的函数
        :param df_train: pandas.DataFrame           需要计算的df
        :param drop_cols: list                      default:["uid", "umobile", "date"];忽略计算的列名
        :param bad: str                             default:"1";响应列的坏标签
        :param good: str                            default:"0";响应列的好标签
        :param response: str                        default:"code";响应列列名
        :param replace_options: dict                default:None;支持对不同列的变量的替换替换
            如对列"i_province2"中的"GS"替换为"CN";对列"i_test"中的-1111替换为-999
            {
            "i_province2":{"GS":"CN"},
            "i_test":{-1111:-999}
            }
            优先级大于replace_all字段
        :param replace_all:dict                     default:None;全局替换变量{key:value}用value替换key

        :param cut_method: str                      default:"cumsum";['quantile', 'cumsum','bestks']可选
        :param max_cut_part: int                    default:20;最大分组数
        :param min_group_num: int                   default:1；最小分组数，包含空值列
        :param group_min_percent: float             default:0.05;每组的最小占比
        :param check_monotonicity: bool             default:True;是否检查单调性
        :param strict_monotonicity: bool            default:True;是否严格单调
        :param fill_value: str                      default:"-999";空值的填充值

        :param enable_iv_threshold: bool            default:True;是否启用iv筛选
        :param iv_min: float                        default:0.02;保留iv大于iv_min的变量；未完成

        :param enable_single_threshold: bool        default:True;是否进行单一性筛选
        :param single_threshold: float              default:0.8;保留单一性小于single_threshold的变量
        :param include_none: bool                   default:True;计算单一性是否包含空值；未完成
        :param n_jobs: int                          default:4;进程数
        :param keep_range: int                      default:500;iv排序后保留的数量
        :return:
        """
        self.df_train = df_train
        self.df_test = df_test
        self.bad = bad
        self.good = good
        self.cut_method = cut_method
        self.max_cut_part = max_cut_part
        self.min_group_num = min_group_num
        self.group_min_percent = group_min_percent
        self.fill_value = fill_value
        self.response = response
        self.enable_iv_threshold = enable_iv_threshold
        self.iv_min = iv_min
        self.enable_single_threshold = enable_single_threshold
        self.single_threshold = single_threshold
        self.include_none = include_none
        self.n_jobs = n_jobs
        self.keep_range = keep_range
        self.strict_monotonicity = strict_monotonicity
        self.woe_fill = woe_fill
        self.fill_items = [-np.inf, np.inf, np.nan]
        if drop_cols is None:
            self.drop_cols = ["uid", "umobile", "date"]
        else:
            self.drop_cols = list(set(["uid", "umobile", "date"] + drop_cols))
        self.sheet1_cols = ['var', 'iv', 'ks', 'group_num', 'missing_percent', 'max_single_percent',
                            'min', '25%', '50%', '75%', 'max', 'std', 'mean', 'mode', 'median']
        self.sheet2_cols = ['Bad_count', 'Good_count', 'IV', 'KS', 'WOE',
                            'cum_bad_percent', 'cum_good_percent', 'default_percent',
                            'inside_bad_percent', 'inside_good_percent',
                            'total_percent_x', 'total_x', 'var_name', 'var_new_x', 'var_scope']
        if df_test is not None:
            self.sheet1_cols.insert(3, 'psi')
            self.sheet2_cols.insert(4, 'PSI')
            self.sheet2_cols.append('var_new_y')
            self.sheet2_cols.append('total_y')
            self.sheet2_cols.append('total_percent_y')
        if isinstance(replace_options, dict):
            self.df_train.replace(replace_options)
            self.df_test = self.df_test.replace(replace_options) if self.df_test is not None else self.df_test
        if isinstance(replace_all, dict):
            self.df_train.replace(replace_all)
            self.df_test = self.df_test.replace(replace_all) if self.df_test is not None else self.df_test

    def get_sheet1_and_sheet2(self):
        df_train = self.df_train.drop(self.drop_cols, axis=1)
        df_train_num = df_train.select_dtypes(exclude=["object"])  # 挑选数值型变量
        df_train_obj = df_train.select_dtypes(include=["object"])  # 挑选字符型变量
        if self.response not in df_train_num:  # 将标签组填充
            df_train_num[self.response] = df_train[self.response]
        if self.response not in df_train_obj:  # 将标签组填充
            df_train_obj[self.response] = df_train[self.response]

        asC_cols = list(df_train_num.drop(self.response, axis=1).columns)
        asD_cols = list(df_train_obj.drop(self.response, axis=1).columns)
        asC_results = self._bin_asC(asC_cols)
        asC_sheet1, asC_sheet2, asC_bins = self._merge_asC(asC_results)
        asD_sheet1, asD_sheet2, asD_bins = self._bin_merge_asD(df_train_obj, asD_cols)
        self.asC = (asC_sheet1, asC_sheet2, asC_bins)
        self.asD = (asD_sheet1, asD_sheet2, asD_bins)
        return ((asC_sheet1, asC_sheet2, asC_bins), (asD_sheet1, asD_sheet2, asD_bins))

    def local_info_save(self, filename, asC=None, asD=None):
        """不需要后缀的名字,保存到本地info表
        """
        info_name = "info_" + filename + '.xlsx'  # excel名字
        if asC and asD:
            sheet1, sheet2, asC_bins = asC
            asD_sheet1, asD_sheet2, asD_bins = asD
        else:
            try:
                sheet1, sheet2, asC_bins = self.asC
                asD_sheet1, asD_sheet2, asD_bins = self.asD
            except Exception as e:
                self.get_sheet1_and_sheet2()
                sheet1, sheet2, asC_bins = self.asC
                asD_sheet1, asD_sheet2, asD_bins = self.asD

        sheet1.update(asD_sheet1)
        sheet2.update(asD_sheet2)
        page1 = pd.concat(sheet1.values())
        page1 = page1.sort_values('iv', ascending=False).iloc[:self.keep_range, :]
        page1 = page1[self.sheet1_cols]
        # 保存到excel
        writer = pd.ExcelWriter(info_name, engine='xlsxwriter')
        page1.to_excel(writer, sheet_name="summery", index=False)
        startrow = 0
        for name in page1.index:
            df_ = sheet2[name]
            df_.to_excel(writer, sheet_name='detail', startrow=startrow, startcol=0)
            startrow = startrow + df_.shape[0] + 2
        writer.save()
        log("结果已保存到:{0}".format(info_name))

    def local_save_m_t(self, filename, asC=None, asD=None):
        """本地保存M表和T表
        """
        now = datetime.datetime.now().strftime("%H%M%S")
        train_name = "M_Data_" + filename + "_" + now + ".csv"
        test_name = "T_Data_" + filename + "_" + now + ".csv"
        if asC and asD:
            sheet1, sheet2, new_bins = asC
            asD_sheet1, asD_sheet2, asD_bins = asD
        else:
            try:
                sheet1, sheet2, new_bins = self.asC
                asD_sheet1, asD_sheet2, asD_bins = self.asD
            except Exception as e:
                self.get_sheet1_and_sheet2()
                sheet1, sheet2, new_bins = self.asC
                asD_sheet1, asD_sheet2, asD_bins = self.asD

        sheet1.update(asD_sheet1)
        sheet2.update(asD_sheet2)
        new_bins.update(asD_bins)
        page1 = pd.concat(sheet1.values())
        page1 = page1.sort_values('iv', ascending=False).iloc[:self.keep_range, :]
        page1 = page1[self.sheet1_cols]
        original_name = page1.index  # 原始名称
        suffix_name = page1['var'].values  # 带后缀的名称

        def __get_d_woe_dict(detail_):  # 字符串型数据进行woe填充的字典
            tmp_dict = detail_.set_index("var_scope")["WOE"].to_dict()
            r_dict = {}
            for k, v in tmp_dict.iteritems():
                if '|' not in k:
                    r_dict[k] = v
                else:
                    for _ in k.split('|'):
                        r_dict[_] = v
            return r_dict

        def __get_d_bin_dict(detail_):  # 字符串型数据进行分箱填充的字典
            tmp_dict = detail_.set_index("var_scope")["var_new_x"].to_dict()
            r_dict = {}
            for k, v in tmp_dict.iteritems():
                if '|' not in k:
                    r_dict[k] = v
                else:
                    for _ in k.split('|'):
                        r_dict[_] = v
            return r_dict

        if self.df_train is not None:  # 保存M表格
            log("开始保存训练集数据:{0}".format(train_name))
            woe_df = pd.DataFrame()
            for i in range(original_name.shape[0]):  # 保存训练集的woe填充
                o_name = original_name[i]
                s_name = suffix_name[i]
                detail_ = sheet2[o_name]
                s_ = self.df_train[o_name]  # 原始的值
                if 'asC' in s_name:  # 如果是连续型
                    woe_dict = detail_.set_index("var_scope")["WOE"].to_dict()
                    s_ = pd.cut(s_, new_bins[o_name]).cat.add_categories([self.fill_value]).fillna(self.fill_value)
                    woe_item = s_.map(woe_dict)
                    bin_item = s_
                elif 'asD' in s_name:  # 如果是字符串型
                    woe_dict = __get_d_woe_dict(detail_)
                    bin_dict = __get_d_bin_dict(detail_)
                    woe_item = s_.map(woe_dict)
                    bin_item = s_.map(bin_dict)
                else:
                    raise ValueError("值{0}类型错误".format(s_name))
                woe_df.insert(0, "WOE_" + s_name, woe_item)
                woe_df.insert(woe_df.shape[1], s_name, bin_item)
            woe_df.to_csv(train_name, index=False)

        if self.df_test is not None:  # 是否有测试集
            log("开始保存测试集数据:{0}".format(test_name))
            woe_df = pd.DataFrame()
            for i in range(original_name.shape[0]):  # 保存训练集的woe填充
                o_name = original_name[i]
                s_name = suffix_name[i]
                detail_ = sheet2[o_name]
                s_ = self.df_test[o_name]  # 原始的值
                if 'asC' in s_name:  # 如果是连续型
                    woe_dict = detail_.set_index("var_scope")["WOE"].to_dict()
                    s_ = pd.cut(s_, new_bins[o_name]).cat.add_categories([self.fill_value]).fillna(self.fill_value)
                    woe_item = s_.map(woe_dict)
                    bin_item = s_
                elif 'asD' in s_name:  # 如果是字符串型
                    woe_dict = __get_d_woe_dict(detail_)
                    bin_dict = __get_d_bin_dict(detail_)
                    woe_item = s_.map(woe_dict)
                    bin_item = s_.map(bin_dict)
                else:
                    raise ValueError("值{0}类型错误".format(s_name))
                woe_df.insert(0, "WOE_" + s_name, woe_item)
                woe_df.insert(woe_df.shape[1], s_name, bin_item)
            woe_df.to_csv(test_name, index=False)  # 保存T表格

    def _bin_asC(self, columns):
        """对连续性变量分箱
        :param columns: 需要分箱的列名
        :return: {name:(bins,df),name:(bins,df)}
        """
        log("开始进行连续性变量分箱操作:{0}".format(self.cut_method))
        binning = Binning(cut_method=self.cut_method, strict_monotonicity=self.strict_monotonicity,
                          min_proportion=self.group_min_percent, bad=self.bad, good=self.good, max_cut_part=self.max_cut_part,
                          fill_value=self.fill_value)
        results = binning.fit(self.df_train, self.df_train[self.response], columns=columns, n_jobs=self.n_jobs)
        return results

    def _merge_asC(self, results):
        """聚合连续性变量的分箱结果
        :return: sheet1,sheet2
        """
        log("分箱结束!开始进行连续性数据聚合...")
        sheet1 = {}
        sheet2 = {}
        new_bins = {}
        for name, value in results.iteritems():  # name:变量名,value: ([分箱的点],分箱完的DataFrame)
            if value is False:  # 如果分箱失败
                continue
            df_ = value[1].sort_index()  # 添加到sheet2当中
            s_ = self.df_train[name]
            df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
            df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()

            if self.enable_single_threshold is True:  # 单一性验证
                if df_["total_percent_x"].max() > self.single_threshold:
                    continue

            if self.enable_iv_threshold is True:  # iv筛选
                if df_["IV"].replace([np.inf, -np.inf], 0).sum() < self.iv_min:
                    continue

            if df_.shape[0] < self.min_group_num:  # 最小分组筛选
                continue
            asC_name = name + "_asC"
            # 计算第二张表
            df_["var_name"] = [asC_name] * df_.shape[0]
            df_["var_new_x"] = df_.index
            df_[asC_name] = df_["var_new_x"].cat.codes
            df_.set_index(asC_name, inplace=True)
            # 最大和最小值修改为极大极小
            cut_points = value[0]
            cut_points[0] = -np.inf
            cut_points[-1] = np.inf
            new_bins[name] = cut_points  # 替换原先没有无穷的切割点
            if self.df_test is not None:  # todo:cut_point有待确定
                tmp_c = pd.cut(self.df_test[name], cut_points)
            else:
                tmp_c = pd.cut(s_, cut_points)

            if len(tmp_c.cat.categories) + 1 == df_.shape[0]:
                tmp_c = tmp_c.cat.add_categories([self.fill_value]).fillna(self.fill_value)

            df_["var_scope"] = tmp_c.cat.categories
            # 挑选指定顺序的列
            if self.df_test is not None:
                tmp_c = tmp_c.to_frame()
                tmp_c.columns = ['bin']
                tmp_c[self.response] = self.df_test[self.response]
                tmp_c["_tmp_count_"] = np.ones(tmp_c.shape[0])  # 给定一个具体的个数的1，可以指定数据类型
                tmp_pivot_df = pd.pivot_table(tmp_c, index='bin', columns=self.response, aggfunc=len, values="_tmp_count_").fillna(0)
                tmp_pivot_df['total'] = tmp_pivot_df[self.good] + tmp_pivot_df[self.bad]
                df_['total_y'] = tmp_pivot_df['total'].values
                df_['total_percent_y'] = df_['total_y'].div(df_['total_y'].sum())
                df_['PSI'] = df_["total_percent_x"].div(df_["total_percent_y"]).map(np.log).mul(df_["total_percent_x"].sub(df_["total_percent_y"]))
                df_['var_new_y'] = df_['var_new_x']

            sheet2[name] = df_[self.sheet2_cols].replace([np.inf, -np.inf, np.nan], 0)
            # 计算第一张表
            r = s_.describe().to_frame().T  # 添加到sheet1当中
            r["iv"] = df_["IV"].replace([np.inf, -np.inf], 0).sum()
            r["ks"] = df_["KS"].max()
            r["group_num"] = df_.shape[0]
            r["missing_percent"] = 1 - r["count"] / s_.shape[0]
            r["max_single_percent"] = (df_["total_x"] / df_["total_x"].sum()).max()
            r["mode"] = ','.join(s_.mode().astype(str).values)  # s_.mode()
            r["median"] = s_.median()
            r["var"] = asC_name
            if self.df_test is not None:
                r["psi"] = df_['PSI'].replace([np.inf, -np.inf], 0).sum()

            sheet1[name] = r[self.sheet1_cols]
        return sheet1, sheet2, new_bins

    def _bin_merge_asD(self, df_train_obj, columns):
        log("开始数值型变量分箱操作")
        sheet1 = {}
        sheet2 = {}
        new_bins = {}
        # df_train_obj[self.response] = df_train_obj[self.response].replace(self.good, "Good_count").replace(self.bad, "Bad_count")
        for name in columns:  # i_province2
            if self.enable_single_threshold is True:  # 单一值筛选
                max_count = float(df_train_obj.groupby(name)[self.response].count().max())
                if max_count / df_train_obj.shape[0] > self.single_threshold:
                    continue
            # 计算每个类型的占比，计算ks，iv等信息
            tmp_c = _Basic.basic_prepare(df_train_obj[self.response], bad=self.bad, good=self.good)
            tmp_c[name] = df_train_obj[name]
            tmp_c = _Basic.get_pivot_table(tmp_c, response=self.response, column=name)
            tmp_c = _Basic.add_basic_info_to_df(tmp_c)
            tmp_c = _Basic.add_woe_iv_to_df(tmp_c)
            if self.woe_fill == 'avg':  # 填充woe的值
                avg = tmp_c["WOE"].replace(self.fill_items, np.nan).mean()
                tmp_c["WOE"] = tmp_c["WOE"].replace(self.fill_items, avg)
            tmp_c = _Basic.add_ks_to_df(tmp_c)
            # 利用上面生成的woe进行分箱,挑选iv最大的组
            parts = []
            last_cut_points = []
            max_iv = 0
            parts, _ = _CutMethods.quantile_cut_flow(tmp_c['WOE'], parts, max_cut_part=self.max_cut_part)
            parts, _ = _CutMethods.cumsum_cut_flow(tmp_c['WOE'], parts, add_min_group=True, max_cut_part=self.max_cut_part)
            for part in parts:  # 目前是取iv最大的组，未来可以加取分组数最多的组
                bins_ = pd.cut(tmp_c['WOE'], part, include_lowest=True)
                tmp_c['bins'] = bins_.cat.codes
                tmp_df = pd.pivot_table(tmp_c, values=['Bad_count', 'Good_count'], index='bins', aggfunc=np.sum)
                tmp_df = _Basic.add_basic_info_to_df(tmp_df)
                tmp_df = _Basic.add_woe_iv_to_df(tmp_df)
                iv_ = tmp_df['IV'].sum()
                if iv_ > max_iv:
                    last_cut_points = part
            # 开始构建sheet1和sheet2
            tmp_c['bins'] = pd.cut(tmp_c['WOE'], last_cut_points, include_lowest=True).cat.codes
            groups = {}
            for i in tmp_c['bins'].unique():
                items_ = tmp_c[tmp_c['bins'] == i].index.tolist()
                groups[i] = items_
            df_ = _Basic.basic_prepare(df_train_obj[self.response], bad=self.bad, good=self.good)
            df_[name] = df_train_obj[name]
            for code, items in groups.iteritems():
                df_[name] = df_[name].replace(items, code)
            df_ = _Basic.get_pivot_table(df_, self.response, name)
            df_ = _Basic.add_basic_info_to_df(df_)
            df_ = _Basic.add_woe_iv_to_df(df_)
            df_ = _Basic.add_ks_to_df(df_)
            df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
            df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()
            # 计算第二张表
            asD_name = name + "_asD"
            df_["var_name"] = [asD_name] * df_.shape[0]
            groups_ = {k: '|'.join(v) for k, v in groups.iteritems()}
            df_["var_new_x"] = pd.Series(groups_)
            df_["var_scope"] = pd.Series(groups_)
            if self.df_test is not None:
                test_df_ = _Basic.basic_prepare(self.df_test[self.response], bad=self.bad, good=self.good)
                test_df_[name] = self.df_test[name]
                for code, items in groups.iteritems():
                    test_df_[name] = test_df_[name].replace(items, code)
                test_df_ = _Basic.get_pivot_table(test_df_, self.response, name)
                test_df_ = _Basic.add_basic_info_to_df(test_df_)
                df_['total_y'] = test_df_['total'].values
                df_['total_percent_y'] = df_['total_y'].div(df_['total_y'].sum())
                df_['PSI'] = df_["total_percent_x"].div(df_["total_percent_y"]).map(np.log).mul(df_["total_percent_x"].sub(df_["total_percent_y"]))
                df_['var_new_y'] = df_['var_new_x']
            new_bins[name] = groups
            sheet2[name] = df_[self.sheet2_cols].replace([np.inf, -np.inf, np.nan], 0)
            # 计算第一张表
            s_ = df_train_obj[name]
            r = s_.describe().to_frame().T  # 添加到sheet1当中
            r["iv"] = df_["IV"].replace([np.inf, -np.inf], 0).sum()
            r["ks"] = df_["KS"].max()
            r["group_num"] = df_.shape[0]
            r["missing_percent"] = 1 - r["count"] / s_.shape[0]
            r["max_single_percent"] = (df_["total_x"] / df_["total_x"].sum()).max()
            r["mode"] = ','.join(s_.mode().astype(str).values)  # s_.mode()
            r["var"] = asD_name
            if self.df_test is not None:
                r["psi"] = df_['PSI'].replace([np.inf, -np.inf], 0).sum()
            try:
                sheet1[name] = r[['var', "iv", 'ks', 'psi', 'group_num', 'missing_percent', 'max_single_percent']]
            except Exception as e:
                sheet1[name] = r[['var', "iv", 'ks', 'group_num', 'missing_percent', 'max_single_percent']]
        return sheet1, sheet2, new_bins
