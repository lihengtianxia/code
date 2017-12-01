# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import datetime
import copy
from bestks.binning import Binning, StrBin
from bestks.basic import _Basic, Progressive

if pd.__version__ < '0.19' or pd.__version__ >= '0.2':
    raise SystemError('pandas version must be 0.19.*,now is "{0}"'.format(pd.__version__))


def show(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


class DcBins(object):
    def __init__(self, df_train, filename, df_test=None,
                 drop_cols=None, bad=1, good=0,
                 replace_options=None, replace_all=None,
                 cut_method="cumsum", max_cut_part=10, min_group_num=3, group_min_percent=0.05,
                 strict_monotonicity=True, best_ks_loop=3,
                 fill_value="-999", response='code', add_info=None,
                 add_min_group=True, keep_separate_value=None,
                 enable_iv_threshold=True, iv_min=0.02, keep_range=500,
                 enable_single_threshold=True, single_threshold=0.8, include_none=True,
                 single_group_num=12, max_of_length=1000, choice_by='iv', woe_inf_fill='avg',
                 save_match=False, save_text=False, save_match_text=False, save_tables_auto=False,
                 n_jobs=4, woe_fill='avg'):
        """
        堡垒机环境下使用的函数
        必填:
        :param df_train: pandas.DataFrame           必填;需要计算的df
        :param filename: str                        保存的文件的主名字,不需要文件后缀名

        :param df_test: pandas.DataFrame            选填;测试集的df
        响应列相关参数
        :param drop_cols: list                      default:["uid", "umobile", "date"];忽略计算的列名
        :param bad: str                             default:1;响应列的坏标签
        :param good: str                            default:0;响应列的好标签
        :param response: str                        default:"code";响应列列名
        :param replace_options: dict                default:None;支持对不同列的变量的替换替换
            如对列"i_province2"中的"GS"替换为"CN";对列"i_test"中的-1111替换为-999
            {
            "i_province2":{"GS":"CN"},
            "i_test":{-1111:-999}
            }
            优先级大于replace_all字段
        :param replace_all:dict                     default:None;全局替换变量{key:value}用value替换key

        数值型分箱参数
        :param cut_method: str                      default:"cumsum";['quantile', 'cumsum','bestks']可选
        :param strict_monotonicity: bool            default:True;是否严格单调
        :param add_min_group: bool                  default:True;cumsum参数, 最小值表现充分时单独一组
        :param keep_separate_value: list            default:None;指定这些值需要单独一组(仅支持这组数据的最大最小值)

        字符串型分箱参数
        :param single_group_num:int                 default:12;种类小于这个数特征，直接按值分箱
        :param max_of_length:int                    default:1000;种类大于这个数的特征，不进行分箱
        :param choice_by:str                        default:'iv';["iv","ks","woe","len"]其中一个，挑选分组时以什么为优先级考虑
        :param woe_inf_fill:str                     default:'avg';填充woe的方法，暂时只支持'avg'

        分箱共享参数
        :param fill_value: str                      default:"-999";空值的填充值
        :param max_cut_part: int                    default:10;最大分组数
        :param min_group_num: int                   default:3；最小分组数，包含空值列
        :param best_ks_loop: int                    default:3;best_ks分箱时的二分次数
        :param group_min_percent: float             default:0.05;每组的最小占比

        iv筛选参数
        :param enable_iv_threshold: bool            default:True;是否启用iv筛选
        :param iv_min: float                        default:0.02;保留iv大于iv_min的变量
        :param keep_range: int                      default:500;iv排序后保留的数量

        单一性筛选参数
        :param enable_single_threshold: bool        default:True;是否进行单一性筛选
        :param single_threshold: float              default:0.8;保留单一性小于single_threshold的变量
        :param include_none: bool                   default:True;计算单一性是否包含空值

        保存内容相关参数
        :param add_info: str or list                default:None;需要添加在M和T表后面的列，如['code','umobile']
        :param save_text: bool                      default:False;是否保持筛选后的区间数据
        :param save_match: bool                     default:False;是否保存筛选后匹配的woe数据
        :param save_match_text: bool                default:True;是否保存筛选后匹配的woe和区间
        :param save_tables_auto: bool               default:False;是否自动保存所有表,local_save_info和local_save_m_t手动保存

        多进程相关
        :param n_jobs: int                          default:4;进程数
        """
        self.df_train = df_train
        self.filename = filename
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
        self.single_group_num = single_group_num
        self.max_of_length = max_of_length
        self.choice_by = choice_by
        self.woe_inf_fill = woe_inf_fill
        self.include_none = include_none
        self.n_jobs = n_jobs
        self.keep_range = keep_range
        self.strict_monotonicity = strict_monotonicity
        self.woe_fill = woe_fill
        self.best_ks_loop = best_ks_loop
        self.add_info = add_info
        self.add_min_group = add_min_group
        self.keep_separate_value = keep_separate_value
        if save_match_text is True:
            self.save_text = True
            self.save_match = True
        else:
            self.save_text = save_text
            self.save_match = save_match
        self.fill_items = [-np.inf, np.inf, np.nan]
        if drop_cols is None:  # 变量挑选时忽略的变量
            self.drop_cols = ["uid", "umobile", "date"]
        else:
            self.drop_cols = list(drop_cols)
        # 写入表中时需要的字段，按写入的顺序排列
        self.sheet1_cols = ['var', 'iv', 'ks', 'group_num', 'missing_percent', 'max_single_percent',
                            'min', '25%', '50%', '75%', 'max', 'std', 'mean', 'mode', 'median']
        self.sheet2_cols = ['Bad_count', 'Good_count', 'IV', 'KS', 'WOE',
                            'cum_bad_percent', 'cum_good_percent', 'default_percent',
                            'inside_bad_percent', 'inside_good_percent',
                            'total_percent_x', 'total_x', 'var_name', 'var_new_x', 'var_scope']
        if df_test is not None:  # 如果有测试集添加测试集的特定字段
            if set(df_train.columns) != set(df_test.columns):
                raise ValueError("df_train.columns != df_test.columns.Plase check it")
            self.sheet1_cols.insert(3, 'psi')
            self.sheet2_cols.insert(4, 'PSI')
            self.sheet2_cols.append('var_new_y')
            self.sheet2_cols.append('total_y')
            self.sheet2_cols.append('total_percent_y')
        if isinstance(replace_options, dict):  # 值替换
            self.df_train.replace(replace_options)
            self.df_test = self.df_test.replace(replace_options) if self.df_test is not None else self.df_test
        if isinstance(replace_all, dict):  # 值替换
            self.df_train.replace(replace_all)
            self.df_test = self.df_test.replace(replace_all) if self.df_test is not None else self.df_test
        # 下面判断输入的参数是否正确
        if self.response in self.drop_cols:  # 如果把标签也舍去了，会把标签从drop中删除
            self.drop_cols.remove(self.response)

        if self.add_info is None:
            check_cols = self.drop_cols
        elif isinstance(self.add_info, str):
            check_cols = self.drop_cols + [self.add_info]
        else:
            check_cols = self.drop_cols + list(self.add_info)
        if set(check_cols) & set(self.df_train.columns) != set(check_cols):  # 如果需要的列不在df中会报错
            raise ValueError('"{0}" not in df_train!'.format(','.join(check_cols)))

        if save_tables_auto:
            self.local_save_info()
            self.local_save_m_t()

    def get_sheet1_and_sheet2(self):
        df_train = self.df_train.drop(self.drop_cols, axis=1)
        df_train_num = df_train.select_dtypes(exclude=["object"])  # 挑选数值型变量
        df_train_obj = df_train.select_dtypes(include=["object"])  # 挑选字符型变量
        if self.response not in df_train_num:  # 将标签组填充
            df_train_num.loc[:, self.response] = df_train[self.response]
        if self.response not in df_train_obj:  # 将标签组填充
            df_train_obj.loc[:, self.response] = df_train[self.response]
        asC_cols = list(df_train_num.drop(self.response, axis=1).columns)
        asD_cols = list(df_train_obj.drop(self.response, axis=1).columns)
        asC_results = self.bin_asC(asC_cols)
        asC_sheet1, asC_sheet2, asC_bins = self.merge_asC(asC_results)
        asD_results = self.bin_asD(asD_cols)
        asD_sheet1, asD_sheet2, asD_bins = self.merge_asD(asD_results)
        self.asC = (asC_sheet1, asC_sheet2, asC_bins)
        self.asD = (asD_sheet1, asD_sheet2, asD_bins)
        return ((asC_sheet1, asC_sheet2, asC_bins), (asD_sheet1, asD_sheet2, asD_bins))

    def local_save_info(self, filename=None, asC=None, asD=None):
        """不需要后缀的名字,保存到本地info表
        """
        if filename is None:
            filename = self.filename
        info_name = "Info_" + filename + '.xlsx'  # excel名字
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
        page1.to_excel(writer, sheet_name="summary", index=False)
        startrow = 0
        for name in page1.index:
            df_ = sheet2[name]
            df_.to_excel(writer, sheet_name='detail', startrow=startrow, startcol=0)
            startrow = startrow + df_.shape[0] + 2
        writer.save()
        show("结果已保存到:{0}".format(info_name))

    def _woe_fill(self, df, sheet2, new_bins, original_name, suffix_name):
        def _get_d_woe_dict(detail_):  # 字符串型数据进行woe填充的字典
            tmp_dict = detail_.set_index("var_scope")["WOE"].to_dict()
            r_dict = {}
            for k, v in tmp_dict.iteritems():
                if '|' not in k:
                    r_dict[k] = v
                else:
                    for _ in k.split('|'):
                        r_dict[_] = v
            return r_dict

        def _get_d_bin_dict(detail_):  # 字符串型数据进行分箱填充的字典
            tmp_dict = detail_.set_index("var_scope")["var_new_x"].to_dict()
            r_dict = {}
            for k, v in tmp_dict.iteritems():
                if '|' not in k:
                    r_dict[k] = v
                else:
                    for _ in k.split('|'):
                        r_dict[_] = v
            return r_dict

        woe_df = pd.DataFrame()
        pg = Progressive(original_name.shape[0], step=2)
        for i in range(original_name.shape[0]):  # 保存训练集的woe填充
            pg.bar(i, "数据填充")
            o_name = original_name[i]  # 原始名字
            s_name = suffix_name[i]  # 加了后缀的名字
            detail_ = sheet2[o_name]  # detail表中的df
            s_ = df[o_name]  # 原始的值
            if 'asC' in s_name:  # 如果是连续型
                # woe_dict = detail_.set_index("var_new_x")["WOE"].to_dict()
                # tmp_s = pd.cut(s_, new_bins[o_name], include_lowest=True).cat.add_categories([self.fill_value]).fillna(self.fill_value)
                woe_dict = detail_.set_index("var_scope")["WOE"].to_dict()
                tmp_s = pd.cut(s_, new_bins[o_name]).cat.add_categories([self.fill_value]).fillna(self.fill_value)
                woe_item = tmp_s.map(woe_dict)
                bin_item = tmp_s
            elif 'asD' in s_name:  # 如果是字符串型
                if new_bins[o_name] == "single":
                    woe_dict = detail_.set_index("var_scope")["WOE"].to_dict()
                    bin_item = s_.fillna(self.fill_value)
                    woe_item = s_.map(woe_dict)
                else:
                    woe_dict = _get_d_woe_dict(detail_)
                    bin_dict = _get_d_bin_dict(detail_)
                    woe_item = s_.map(woe_dict)
                    bin_item = s_.map(bin_dict)
            else:
                raise ValueError("值{0}类型错误".format(s_name))
            if self.save_match is True:
                woe_df.insert(0, "WOE_" + s_name, woe_item)
            if self.save_text is True:
                woe_df.insert(woe_df.shape[1], s_name, bin_item)
        woe_df[self.response] = df[self.response]  # 默认添加code列
        if self.add_info is not None:
            woe_df[self.add_info] = df[self.add_info]
        return woe_df

    def local_save_m_t(self, filename=None, asC=None, asD=None):
        """本地保存M表和T表
        """
        if filename is None:
            filename = self.filename
        train_name = "M_Data_" + filename + ".csv"
        test_name = "T_Data_" + filename + ".csv"
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

        if self.df_train is not None:  # 保存M表格
            show("开始保存训练集数据:{0}".format(train_name))
            woe_df = self._woe_fill(self.df_train, sheet2, new_bins, original_name, suffix_name)
            woe_df.to_csv(train_name, index=False)

        if self.df_test is not None:  # 是否有测试集
            show("开始保存测试集数据:{0}".format(test_name))
            woe_df = self._woe_fill(self.df_test, sheet2, new_bins, original_name, suffix_name)
            woe_df.to_csv(test_name, index=False)  # 保存T表格

    def bin_asC(self, columns):
        """对连续性变量分箱
        :param columns: 需要分箱的列名
        :return: {name:(bins,df),name:(bins,df)}
        """
        show("开始进行连续性变量分箱操作:{0}".format(self.cut_method))
        binning = Binning(cut_method=self.cut_method, strict_monotonicity=self.strict_monotonicity,
                          group_min_percent=self.group_min_percent,
                          bad=self.bad, good=self.good,
                          max_cut_part=self.max_cut_part, min_cut_part=self.min_group_num,
                          fill_value=self.fill_value, loop=self.best_ks_loop,
                          add_min_group=self.add_min_group, keep_separate_value=self.keep_separate_value
                          )
        results = binning.fit(self.df_train, self.df_train[self.response], columns=columns, n_jobs=self.n_jobs)
        return results

    def merge_asC(self, results):
        """聚合连续性变量的分箱结果
        :return: sheet1,sheet2
        """
        show("分箱结束!开始进行连续性数据聚合...")
        sheet1 = {}
        sheet2 = {}
        new_bins = {}
        pg = Progressive(len(results), step=2)
        speed_of_progress = 0
        for name, value in results.iteritems():  # name:变量名,value: ([分箱的点],分箱完的DataFrame)
            pg.bar(speed_of_progress, msg="连续性数据聚合")
            speed_of_progress += 1
            if value is False:  # 如果分箱失败
                continue
            df_ = value[1].sort_index()  # 添加到sheet2当中
            s_ = self.df_train[name]
            df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
            df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()

            if self.enable_single_threshold is True:  # 单一性验证
                if self.include_none is False:  # 如果不包含缺失值
                    try:
                        single_max = df_["total_percent_x"].drop(self.fill_value).max()
                    except ValueError as e:  # 没有缺失列
                        single_max = df_["total_percent_x"].max()
                else:
                    single_max = df_["total_percent_x"].max()
                if single_max > self.single_threshold:
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
            new_bins[name] = copy.deepcopy(cut_points)
            if self.df_test is not None:
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

    def bin_asD(self, columns):
        show("开始进行数值型变量分箱操作")
        binning = StrBin(fill_value=self.fill_value, good=self.good, bad=self.bad,
                         group_min_percent=self.group_min_percent, choice_by=self.choice_by,
                         single_group_num=self.single_group_num, max_of_length=self.max_of_length,
                         woe_inf_fill=self.woe_inf_fill, max_cut_part=self.max_cut_part)
        results = binning.fit(self.df_train, self.df_train[self.response], columns=columns, n_jobs=self.n_jobs)
        return results

    def merge_asD(self, results):
        show("分箱结束!开始进行数值型数据聚合...")
        sheet1 = {}
        sheet2 = {}
        new_bins = {}
        pg = Progressive(len(results), step=2)
        speed_of_progress = 0
        for name, value in results.iteritems():  # name:变量名,value: ([分箱的点],分箱完的DataFrame)
            pg.bar(speed_of_progress, "数值型数据聚合")
            speed_of_progress += 1
            df_ = value[1]
            s_ = self.df_train[name]
            asD_name = name + "_asD"
            if value[0] is False:  # 如果分箱失败
                continue
            elif value[0] == 'single':
                df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
                df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()
                if self.enable_single_threshold is True:  # 单一性验证
                    if df_["total_percent_x"].max() > self.single_threshold:
                        continue
                if self.enable_iv_threshold is True:  # iv筛选
                    if df_["IV"].replace([np.inf, -np.inf], 0).sum() < self.iv_min:
                        continue
                df_["var_name"] = [asD_name] * df_.shape[0]
                df_["var_new_x"] = df_.index
                df_["var_scope"] = df_.index
                df_[asD_name] = range(0, df_.shape[0])
                df_.set_index(asD_name, inplace=True)
                if self.df_test is not None:
                    tmp_df = s_.to_frame().fillna(self.fill_value)
                    tmp_df["_tmp_count_"] = np.ones(tmp_df.shape[0])
                    tmp_df = tmp_df.groupby(name).count()
                    tmp_df.columns = ['total_y']
                    df_ = pd.concat([df_, tmp_df], axis=1)
                    df_['total_percent_y'] = df_['total_y'].div(df_['total_y'].sum())
                    df_['PSI'] = df_["total_percent_x"].div(df_["total_percent_y"]).map(np.log).mul(df_["total_percent_x"].sub(df_["total_percent_y"]))
                    df_['var_new_y'] = df_['var_new_x']
                    sheet2[name] = df_[self.sheet2_cols].replace([np.inf, -np.inf, np.nan], 0)
                new_bins[name] = "single"
                sheet2[name] = df_[self.sheet2_cols].replace([np.inf, -np.inf, np.nan], 0)
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

            elif isinstance(value[0], dict):
                groups = value[0]
                df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
                df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()
                # if self.enable_single_threshold is True:  # 单一性验证
                #     if df_["total_percent_x"].max() > self.single_threshold:
                #         continue
                if self.enable_iv_threshold is True:  # iv筛选
                    if df_["IV"].replace([np.inf, -np.inf], 0).sum() < self.iv_min:
                        continue
                # 计算第二张表
                df_["var_name"] = [asD_name] * df_.shape[0]
                groups_ = {k: '|'.join(v) for k, v in groups.iteritems()}
                df_["var_new_x"] = pd.Series(groups_)
                df_["var_scope"] = pd.Series(groups_)
                df_.index.name = asD_name
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
                    sheet1[name] = r[['var', "iv", 'ks', 'psi', 'group_num', 'missing_percent', 'max_single_percent']]
                else:
                    sheet1[name] = r[['var', "iv", 'ks', 'group_num', 'missing_percent', 'max_single_percent']]
            else:
                raise ValueError("分箱返回的结果不符合预期{0}".format(type(value[0])))

        return sheet1, sheet2, new_bins
