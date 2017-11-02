# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import datetime
import time
from bestks.binning import Binning
from bestks.basic import _Basic


def log(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


def save_to_excel(page1, sheet2, filename):
    """保存info信息到excel
    """
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    page1.to_excel(writer, sheet_name="summery", index=False)
    startrow = 0
    for name in page1.index:
        df_ = sheet2[name]
        df_.to_excel(writer, sheet_name='detail', startrow=startrow, startcol=0)
        startrow = startrow + df_.shape[0] + 2
    writer.save()


def fill_woe_bin(df_train, page1, sheet2, new_bins, fill_value,
                 train_filename, test_filename, df_test=None):
    """woe和分箱填充数据
    """
    woe_df = pd.DataFrame()
    for name in page1.index:
        value = sheet2[name]
        woe_dict = value.set_index("var_scope")["WOE"].to_dict()
        s_ = df_train[name]
        s_ = pd.cut(s_, new_bins[name]).cat.add_categories([fill_value]).fillna(fill_value)
        woe_df.insert(0, "WOE_" + name + "_asC", s_.map(woe_dict))
        woe_df.insert(woe_df.shape[1], name + "_asC", s_)
    woe_df.to_csv(train_filename, index=False)

    if df_test is not None:  # 是否有测试集
        woe_df = pd.DataFrame()
        for name in page1.index:
            value = sheet2[name]
            woe_dict = value.set_index("var_scope")["WOE"].to_dict()
            s_ = df_test[name]
            s_ = pd.cut(s_, new_bins[name]).cat.add_categories([fill_value]).fillna(fill_value)
            woe_df.insert(0, "WOE_" + name + "_asC", s_.map(woe_dict))
            woe_df.insert(woe_df.shape[1], name + "_asC", s_)
        woe_df.to_csv(test_filename, index=False)


def get_sheet1_and_sheet2(df_train, df_test=None,
                          drop_cols=None, bad=1, good=0,
                          cut_method="cumsum", max_cut_part=10, min_group_num=3, group_min_percent=0.05,
                          check_monotonicity=True, strict_monotonicity=True,
                          fill_value="-999", response='code',
                          enable_iv_threshold=True, iv_min=0.02, keep_largest=2000,
                          enable_single_threshold=True, single_threshold=0.8, include_none=True,
                          n_jobs=4):
    """
    对数据进行合并，生成sheet1和sheet2用来生成excel和csv
    """
    if drop_cols is None:
        drop_cols = ["uid", "umobile", "date"]
    else:
        drop_cols = list(set(["uid", "umobile", "date"] + drop_cols))
    df_train = df_train.drop(drop_cols, axis=1)
    df_train_num = df_train.select_dtypes(exclude=["object"])
    if response not in df_train_num:
        df_train_num[response] = df_train[response]
    df_train_obj = df_train.select_dtypes(include=["object"])
    if response not in df_train_obj:
        df_train_obj[response] = df_train[response]
    num_cols = list(df_train_num.drop(response, axis=1).columns)
    sheet1_cols = ['var', 'iv', 'ks', 'group_num', 'missing_percent', 'max_single_percent',
                   'min', '25%', '50%', '75%', 'max', 'std', 'mean', 'mode', 'median']
    sheet2_cols = ['Bad_count', 'Good_count', 'IV', 'KS', 'WOE',
                   'cum_bad_percent', 'cum_good_percent', 'default_percent',
                   'inside_bad_percent', 'inside_good_percent',
                   'total_percent_x', 'total_x', 'var_name', 'var_new_x', 'var_scope']
    if df_test is not None:
        sheet1_cols.insert(3, 'psi')
        sheet2_cols.insert(4, 'PSI')
        sheet2_cols.append('var_new_y')
        sheet2_cols.append('total_y')
        sheet2_cols.append('total_percent_y')
    log("开始进行连续性变量分箱操作:{0}".format(cut_method))
    binning = Binning(cut_method=cut_method, strict_monotonicity=strict_monotonicity,
                      min_proportion=group_min_percent, bad=bad, good=good, max_cut_part=max_cut_part,
                      fill_value=fill_value)
    results = binning.fit(df_train, df_train[response], columns=num_cols, n_jobs=n_jobs)
    # results 结构为 {name:(bins,df),name:(bins,df)}
    log("分箱结束!开始进行连续性数据聚合...")
    sheet1 = {}
    sheet2 = {}
    new_bins = {}
    for name, value in results.iteritems():  # name:变量名,value: ([分箱的点],分箱完的DataFrame)
        if value is False:  # 如果分箱失败
            continue
        df_ = value[1].sort_index()  # 添加到sheet2当中
        s_ = df_train[name]
        df_["total_x"] = df_["Bad_count"] + df_["Good_count"]
        df_["total_percent_x"] = df_["total_x"] / df_["total_x"].sum()

        if enable_single_threshold is True:  # 单一性验证
            if df_["total_percent_x"].max() > single_threshold:
                continue

        if enable_iv_threshold is True:  # iv筛选
            if df_["IV"].replace([np.inf, -np.inf], 0).sum() < iv_min:
                continue

        if df_.shape[0] < min_group_num:  # 最小分组筛选
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
        if df_test is not None:
            tmp_c = pd.cut(df_test[name], cut_points)
        else:
            tmp_c = pd.cut(s_, cut_points)

        if len(tmp_c.cat.categories) + 1 == df_.shape[0]:
            tmp_c = tmp_c.cat.add_categories([fill_value]).fillna(fill_value)

        df_["var_scope"] = tmp_c.cat.categories
        # 挑选指定顺序的列
        if df_test is not None:
            tmp_c = tmp_c.to_frame()
            tmp_c.columns = ['bin']
            tmp_c[response] = df_test[response]
            tmp_c["_tmp_count_"] = np.ones(tmp_c.shape[0])  # 给定一个具体的个数的1，可以指定数据类型
            tmp_pivot_df = pd.pivot_table(tmp_c, index='bin', columns=response, aggfunc=len, values="_tmp_count_").fillna(0)
            tmp_pivot_df['total'] = tmp_pivot_df[good] + tmp_pivot_df[bad]
            df_['total_y'] = tmp_pivot_df['total'].values
            df_['total_percent_y'] = df_['total_y'].div(df_['total_y'].sum())
            df_['PSI'] = df_["total_percent_x"].div(df_["total_percent_y"]).map(np.log).mul(df_["total_percent_x"].sub(df_["total_percent_y"]))
            df_['var_new_y'] = df_['var_new_x']

        sheet2[name] = df_[sheet2_cols].replace([np.inf, -np.inf, np.nan], 0)
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
        if df_test is not None:
            r["psi"] = df_['PSI'].replace([np.inf, -np.inf], 0).sum()

        sheet1[name] = r[sheet1_cols]

    log("开始稀疏变量分箱操作")
    df_train_obj[response] = df_train_obj[response].replace(good, "Good_count").replace(bad, "Bad_count")
    for name in df_train_obj.columns:  # i_province2
        if enable_single_threshold is True:  # 单一值筛选
            max_count = float(df_train_obj.groupby(name)[response].count().max())
            if max_count / df_train_obj.shape[0] > single_threshold:
                continue
        tmp_c = _Basic.basic_prepare(df_train_obj[response], bad=bad, good=good)
        tmp_c[name] = df_train_obj[name]
        tmp_c = _Basic.get_pivot_table(tmp_c, response=response, column=name)
        tmp_c = _Basic.add_basic_info_to_df(tmp_c)
        tmp_c = _Basic.add_woe_iv_to_df(tmp_c)
        tmp_c = _Basic.add_ks_to_df(tmp_c)
        print(name)
    # todo:字符串变量筛选(如省份)

    return sheet1, sheet2, new_bins


def logic_flow(df_train, filename, df_test=None,
               drop_cols=None, bad=1, good=0,
               cut_method="cumsum", max_cut_part=10, min_group_num=3, group_min_percent=0.05,
               check_monotonicity=True, strict_monotonicity=True,
               fill_value="-999", response='code',
               enable_iv_threshold=True, iv_min=0.02,
               enable_single_threshold=True, single_threshold=0.8, include_none=True,
               n_jobs=4, keep_range=500):
    """
    堡垒机环境下使用的函数
    :param df_train: pandas.DataFrame           需要计算的df
    :param filename: str                        输出文件的文件名,不需要后缀
    :param drop_cols: list                      default:["uid", "umobile", "date"];忽略计算的列名
    :param bad: str                             default:"1";响应列的坏标签
    :param good: str                            default:"0";响应列的好标签
    :param response: str                        default:"code";响应列列名


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
    now = datetime.datetime.now().strftime("%H%M%S")
    info_name = "info_" + filename + '.xlsx'
    train_name = "M_Data_" + filename + "_" + now + ".csv"
    test_name = "T_Data_" + filename + "_" + now + ".csv"
    start = time.time()
    sheet1, sheet2, new_bins = get_sheet1_and_sheet2(df_train, df_test=df_test,
                                                     drop_cols=drop_cols, bad=bad, good=good,
                                                     cut_method=cut_method, max_cut_part=max_cut_part, min_group_num=min_group_num, group_min_percent=group_min_percent,
                                                     check_monotonicity=check_monotonicity, strict_monotonicity=strict_monotonicity,
                                                     fill_value=fill_value, response=response,
                                                     enable_iv_threshold=enable_iv_threshold, iv_min=iv_min,
                                                     enable_single_threshold=enable_single_threshold, single_threshold=single_threshold, include_none=include_none,
                                                     n_jobs=n_jobs)
    page1 = pd.concat(sheet1.values())
    page1 = page1.sort_values('iv', ascending=False).iloc[:keep_range, :]
    log("保存到文件{0}".format(filename))
    save_to_excel(page1=page1, sheet2=sheet2, filename=info_name)
    log("开始woe替换数据")
    fill_woe_bin(df_train=df_train, page1=page1, sheet2=sheet2, new_bins=new_bins, fill_value=fill_value,
                 train_filename=train_name, test_filename=test_name, df_test=df_test)
    log("耗时:{0}秒".format(int(time.time() - start)))


def dc_logic_flow():
    """在dc平台上使用的函数
    需要先创建T_Data_{0},M_Data_{0},info_Data_{0}这三个表格
    可能还需要生成的detail这个excel.
    """
    pass


if __name__ == '__main__':
    df = pd.read_csv("data/mogu1013.csv")
    df_train = df.sample(n=int(df.shape[0] * 0.7))
    df_test = df.sample(n=int(df.shape[0] * 0.3))
    logic_flow(df, 'mogu-test', df_test=df_test, drop_cols=['userid'], cut_method="cumsum")
