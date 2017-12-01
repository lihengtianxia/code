# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
"""
   # 参数
   # ----------
   # output_name                 必填，需要以.xlsx结尾，填写规范见补充

   # 模型参数
   # filter_var                  选填，默认True，变量筛选
   # response_name               选填，默认"code"，响应的列名
   # model_type                  选填，默认"sm"，为传统评分卡，可选"ml"
   # pvalue_limit                选填，默认0.2，type为"sm"时生效，stepwise时的pvalue的limit
   # enable_model_cv             选填，默认False，type为"ml"时生效，ml中l2时启用cv来获取参数
   # enable_negative             选填，默认True为负，控制coef方向，False为正，None不做限制
   # exclude_column              选填，默认None，建模时指定排除的column的内容，支持str和list

   # 相关性筛选
   # enable_corr_filter          选填，默认True，是否启用相关性筛选
   # corr_limit                  选填，默认0.7，在启用相关性筛选时的筛选阈值
   # select_by                   选填，默认"iv"，可选"ks", "psi", "ks_gap", "iv_gap"

   # 交叉验证相关
   # enable_train_test           选填，默认False，True时读取T_Data...csv做交叉验证

   # 评分卡相关
   # bad                         选填，默认为1
   # good                        选填，默认为0
   # odds                        选填，默认样本好坏比
   # base_score                  选填，默认580，基础分
   # double_score                选填，默认50，翻倍分
   # round_score                 选填，默认True，是否将评分四舍五入后输出
   # cut_method                  选填，默认"cumsum"，qcut的方法，支持"quantile"
   # display_ks                  选填，默认(10, 0.09)，计算ks时的等分组数和等量占比

   # 输出相关
   # add_info                    选填，默认None，输出添加列，支持str和list
   # save_proba                  选题，默认False，输出计算样本的分数和概率
   # sort_score_by               选填，默认"index"将按区间排序，可选"value"
   # save_model_pkl              选填，默认False，将模型文件存为.pkl输出到本地
   # save_var_dict_pkl           选填，默认(False, 5)，输出可行的变量组合的字典

   # 数据翻译
   # translate_var               选填，默认False，翻译变量
   # var_dict_path               选填，默认"/tmp/var_dict.xlsx"，变量翻译的字典路径
   # var_series_name             选填，默认"long_name"，变量文件字典中待翻译变量的列名称

   # ----------

   # 补充：
   # mc仅支持包含列名中包含"WOE"关键字的数据
   # 通过mc.__plot__来查看和修改绘图相关的参数
   # 数据mc.__data__为None时会尝试读取本地环境的文件
   # 通过mc.__data__来查看和修改测试集和训练集的数据状况
   # sm建模方法的逐步回归会按select_by中的变量顺序来进行
   # add_column和exclude_column必须为所选加载的数据中已有的列
   # filter_var调为False，可在模型变量迁移或交叉验证效果时使用
   # 启用enable_corr_filter时，筛选标准的值相同时结果具有一定的随机性
   # enable_corr_filter为True时，m_data必须None，output_name必须和dc的一致
   # add_info的填充项必须为M/O数据中已有的列名，add_info非None时才输出total_score表
   # enable_negative需要结合response使用，dc产生的M_Data中，默认bad为1，good为0，enable_negative为True
   # ----------

   # 输出
   # ----------
   # 以文件形式存在本地
"""
from __future__ import unicode_literals, print_function, division
import pandas as pd
import numpy as np
import logging
import copy
import os
import shutil
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt
from math import log
import datetime
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.tools as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from bestks.basic import _Basic
from progressive import Progressive


def show(msg):
    date = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(date + "*****" + msg)


def pd_qcut(x, q, **kwargs):
    if pd.__version__ > '0.20':  # 如果pandas版本大于0.20，有duplicates参数去除重复的切割点
        return pd.qcut(x, q, duplicates='drop', **kwargs)
    else:
        try:
            return pd.qcut(x, q, **kwargs)
        except ValueError as e:
            show('Error for qcut, merge cut point...')
            qpoints = np.linspace(0, 1, q)
            values = x.sort_values().values
            length = x.shape[0]
            points = set()
            for i in qpoints:
                num = int(round(i * length)) - 1
                if num < 0:
                    num = 0
                points.add(values[num])
            points = sorted(points)
            return pd.cut(x, points, include_lowest=True, **kwargs)


class ModelSelection(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __get_match_data_m(self):
        woe_col = [i for i in self.kwargs['df_m'] if i.startswith("WOE")]
        woe_total_x = self.kwargs['df_m'][woe_col]
        woe_response = self.kwargs['df_m'][self.kwargs.get('response', 'code')]
        return woe_total_x, woe_response

    def __get_match_data_t(self):
        woe_col = [i for i in self.kwargs['df_t'] if i.startswith("WOE")]
        woe_total_x = self.kwargs['df_t'][woe_col]
        woe_response = self.kwargs['df_t'][self.kwargs.get('response', 'code')]
        return woe_total_x, woe_response

    @staticmethod
    def __selection_by_l1(x, y):
        model = linear_model.LogisticRegression(penalty='l1')
        model.fit(x, y)
        result = pd.DataFrame(
            {'var': x.columns, 'coef': model.coef_[0]})
        return result[(result['coef'] != 0)]['var'].tolist()

    @staticmethod
    def __filter_by_l2(variables, response, select_list):
        while True:
            model = linear_model.LogisticRegression(penalty='l2')
            model.fit(variables[select_list], response)
            if np.all(model.coef_[0] < 0):  # 只取系数是负的,便于解释业务场景
                return variables, response, select_list
            else:
                result = pd.DataFrame(
                    {'var': select_list, 'coef': model.coef_[0]})
                select_list = result[(result['coef'] < 0)]['var'].tolist()

    def __selected_by_sm_logit(self, variables, response, select_list):
        tmp = sm.Logit(
            response, variables[select_list]).fit(disp=False)
        p_series_ = pd.Series(tmp.pvalues)
        p_series_ = p_series_[p_series_ > self.kwargs.get('pvalue_limit', 0.2)].index.tolist()
        if self.kwargs.get('enable_negative', True):
            c_series_ = pd.Series(tmp.params)
            c_series_ = c_series_[c_series_ > 0].index.tolist()
            return [i for i in select_list
                    if i not in p_series_ and i not in c_series_]
        else:
            return [i for i in select_list if i not in p_series_]

    def filter_by_l2(self, x=None, y=None):
        if (x is None) or (y is None):
            show("自动获取x和y!")
            x, y = self.__get_match_data_m()
            if self.kwargs.get('exclude_column', None) is not None:
                x = x.drop(self.kwargs['exclude_column'], axis=1)

        select_cols = self.__selection_by_l1(x, y)
        x, y, select_cols = self.__filter_by_l2(x, y, select_cols)
        x.loc[:, 'Intercept'] = np.ones(x.shape[0])
        select_cols.append('Intercept')
        return x, y, select_cols

    def filter_by_stepwise(self, x=None, y=None):
        if (x is None) or (y is None):
            show("自动获取x和y!")
            x, y = self.__get_match_data_m()
            if self.kwargs.get('exclude_column', None) is not None:
                x = x.drop(self.kwargs['exclude_column'], axis=1)
        x.loc[:, 'Intercept'] = np.ones(x.shape[0])
        columns = self.kwargs['df_i'].sort_values(self.kwargs.get('select_by', 'iv'), ascending=False).index.tolist()
        columns = ["WOE_" + i for i in columns if "WOE_" + i in x.columns]
        select_list = ['Intercept']
        pg = Progressive(len(columns), 2)
        speed_of_progress = 0
        for i in columns:
            pg.bar(speed_of_progress, 'stepwise')
            speed_of_progress += 1
            select_list.append(i)
            try:
                select_list = self.__selected_by_sm_logit(
                    x, y, select_list)
            except (np.linalg.LinAlgError,
                    st.sm_exceptions.PerfectSeparationError) as error:
                logging.error(error)
                select_list.remove(i)
        if 'Intercept' not in select_list:
            select_list.append('Intercept')
        return x, y, select_list


class ModelControl(object):
    # enable_train_test
    model_filter = {}
    train_cut_tb = None
    train_qcut_tb = None
    test_cut_tb = None
    test_qcut_tb = None
    score_basic = {}  # 用来生成score_basic表
    train_auc_param = None  # 用来画auc曲线
    test_auc_param = None  # 用来画auc曲线
    train_proba_score = None  # 用以save_proba参数()
    test_proba_score = None  # 用以save_proba参数()

    def __init__(self, filename, df_m=None, df_t=None, df_i=None, df_i_d=None,
                 add_info=None, save_proba=False, **kwargs):
        kwargs['df_m'] = pd.read_csv('M_Data_' + filename + '.csv') if df_m is None else df_m
        kwargs['df_i'] = pd.read_excel('Info_' + filename + '.xlsx', sheetname='summary', index_col='var') if df_i is None else df_i
        self.df_i_d = pd.read_excel('Info_' + filename + '.xlsx', sheetname='detail') if df_i_d is None else df_i_d
        self.df_m = kwargs['df_m']
        self.df_i = kwargs['df_i']
        if kwargs.get('enable_train_test', False) is True:
            kwargs['df_t'] = pd.read_csv('T_Data_' + filename + '.csv') if df_m is None else df_t
            self.df_t = kwargs['df_t']
        self.kwargs = kwargs
        # 将模型的筛选函数赋值给字典
        ms = ModelSelection(**kwargs)
        self.model_filter['ml'] = ms.filter_by_l2
        self.model_filter['sm'] = ms.filter_by_stepwise
        self.filename = filename
        self.select_by = self.kwargs.get('select_by', 'iv')  # iv,ks,psi
        # 评分卡相关参数
        self.bad = self.kwargs.get('bad', 1)
        self.good = self.kwargs.get('good', 0)
        self.odds = self.kwargs.get('odds', None)
        self.response = self.kwargs.get('response', 'code')
        self.base_score = self.kwargs.get('base_score', 580)
        self.double_score = self.kwargs.get('double_score', 50)
        self.round_score = self.kwargs.get('round_score', True)
        self.cut_method = self.kwargs.get('cut_method', 'qcut')  # cut
        self.display_ks = self.kwargs.get('display_ks', 10)
        self.enable_train_test = self.kwargs.get('enable_train_test', False)
        self.round_score = self.kwargs.get('round_score', True)
        self.corr_limit = self.kwargs.get('corr_limit', 0.7)
        self.add_info = add_info
        self.save_proba = save_proba
        self.exclude_column = self.kwargs.get('exclude_column', None)
        if self.odds is None:
            molecule_ = self.kwargs['df_m'][self.response][self.kwargs['df_m'][self.response] == self.bad].count()
            denominator = self.kwargs['df_m'][self.response][self.kwargs['df_m'][self.response] == self.good].count()
            self.odds = molecule_ / float(denominator)

    @staticmethod
    def append_to_excel(dfs, writer, sheet_name, index=True):
        show('Add to excel:{0}'.format(sheet_name))
        if isinstance(dfs, (list, tuple, set)):
            startrow = 0
            for df in dfs:
                df.to_excel(writer, sheet_name, startrow=startrow, startcol=0, index=index)
                startrow += df.shape[0]
                startrow += 2
        else:
            dfs.to_excel(writer, sheet_name, index=index)

    @staticmethod
    def get_standard_model(x, y, cols):
        use_df = x[cols]
        # model = linear_model.LogisticRegression(penalty='l1')
        # model.fit(use_df, y)
        model = sm.Logit(y, use_df).fit(disp=False)

        return model

    def get_score_card(self, x, cols, model):
        """
        base_score=A-B*log(odds)
        base_score-double_score=A-B*log(2*odds)
        B = double_score / log(2)
        A = base_score + B * log(odds)
        :returns Series(scores),Series(pred_y)
        """
        B = self.double_score / log(2)
        A = self.base_score + B * log(self.odds)
        pred_y = model.predict(x[cols])
        scores = [A - B * log(i / (1 - i)) for i in pred_y]  # 计算总分
        if self.round_score is True:
            scores = [int(round(i)) for i in scores]
        self.score_basic = dict(double_score=self.double_score, basic_b=B, basic_a=A,
                                base_score=self.base_score, odds=self.odds)
        return pd.Series(scores, name='score'), pd.Series(pred_y, name='pred_y')

    @staticmethod
    def _get_info_table(df_, column, response):
        df_ = copy.deepcopy(df_)
        df_ = _Basic.get_pivot_table(df_, column=column, response=response)
        df_ = _Basic.add_basic_info_to_df(df_)
        # df_ = _Basic.add_woe_iv_to_df(df_)
        df_["cum_good_percent"] = df_["Good_count"].cumsum().div(
            np.dot(df_["Good_count"].sum(), np.ones(df_.shape[0])))
        df_["cum_bad_percent"] = df_["Bad_count"].cumsum().div(
            np.dot(df_["Bad_count"].sum(), np.ones(df_.shape[0])))
        df_["KS"] = df_["cum_good_percent"].sub(df_["cum_bad_percent"]).abs()
        return df_

    @staticmethod
    def _get_psi(train_tb, test_tb):
        tmp = pd.concat([train_tb['total_percent'], test_tb['total_percent']], axis=1)
        tmp.columns = ['train', 'test']
        tmp['psi'] = tmp['train'].div(tmp['test']).map(log).mul(tmp["train"].sub(tmp["test"]))
        return tmp['psi'].sum()

    def get_var_summary(self, cols):
        cols = list(set(cols) - {'Intercept'})
        cols = [col[4:] for col in cols]
        return self.df_i.loc[cols, :].sort_values('iv', ascending=False)

    def get_var_detail(self, cols):
        cols = list(set(cols) - {'Intercept'})
        cols = [col[4:] for col in cols]
        dfs = []
        for col in cols:
            tmp = self.df_i_d[self.df_i_d['var_name'] == col]
            new_name = tmp['var_name'].values[0]
            old_name = tmp.iloc[:, 0].name
            tmp = tmp.rename(columns={old_name: new_name})
            dfs.append(tmp)
        return dfs

    def get_score_basic(self):
        data = pd.DataFrame(self.score_basic.values(), index=self.score_basic.keys(), columns=['score'])
        return data

    def get_results_summary(self, model, x, cols):
        scores, pred_y = self.get_score_card(x, cols, model)
        self.train_proba_score = (scores, pred_y)  # 用来显示表格
        y = self.df_m[self.response]
        tmp = _Basic.basic_prepare(y, good=self.good, bad=self.bad)
        tmp.loc[:, 'score'] = scores
        # cut不会有问题
        tmp.loc[:, 'cut'], cut_points = pd.cut(scores, self.display_ks, retbins=True, include_lowest=True)
        # qcut可能出现切割点一样的问题，需要减少个数
        tmp.loc[:, 'qcut'], qcut_points = pd_qcut(scores, self.display_ks, retbins=True)

        if self.round_score is True:  # 如果对切割点四舍五入
            cut_points = [int(round(i)) for i in cut_points]
            cut_points[0] = scores.min()  # 修改最小值为一样
            tmp.loc[:, 'cut'] = pd.cut(scores, cut_points, include_lowest=True)
            qcut_points = [int(round(i)) for i in qcut_points]
            tmp.loc[:, 'qcut'] = pd.cut(scores, qcut_points, include_lowest=True)
        # y_score = model.predict_proba(x[cols])[:, 1]
        y_score = model.predict(x[cols])
        uncut_ks = self._get_info_table(tmp, 'score', self.response)['KS'].max()
        self.train_cut_tb = self._get_info_table(tmp, 'cut', self.response)
        cut_ks = self.train_cut_tb['KS'].max()
        self.train_qcut_tb = self._get_info_table(tmp, 'qcut', self.response)
        qcut_ks = self.train_qcut_tb['KS'].max()
        self.train_auc_param = (y, y_score)  # 为了画曲线
        auc = roc_auc_score(y, y_score)
        ap = average_precision_score(y, y_score)
        data = pd.DataFrame([auc, ap, uncut_ks, cut_ks, qcut_ks], columns=['train'],
                            index=['auc', 'ap', 'uncut_ks', 'cut_ks', 'qcut_ks'])
        if self.enable_train_test:
            if 'Intercept' in cols:
                self.df_t['Intercept'] = np.ones(self.df_t.shape[0])
            scores_, pred_y = self.get_score_card(self.df_t, cols, model)
            self.test_proba_score = (scores_, pred_y)
            x_ = self.df_t[cols]
            y_ = self.df_t[self.response]
            tmp_ = _Basic.basic_prepare(y_, good=self.good, bad=self.bad)
            tmp_.loc[:, 'score'] = scores_
            tmp_.loc[:, 'cut'] = pd.cut(scores_, cut_points, include_lowest=True)
            tmp_.loc[:, 'qcut'] = pd.cut(scores_, qcut_points, include_lowest=True)
            # y_score_ = model.predict_proba(x_)[:, 1]
            y_score_ = model.predict(x_)
            uncut_ks_ = self._get_info_table(tmp_, 'score', self.response)['KS'].max()
            self.test_cut_tb = self._get_info_table(tmp_, 'cut', self.response)
            cut_ks_ = self.test_cut_tb['KS'].max()
            self.test_qcut_tb = self._get_info_table(tmp_, 'qcut', self.response)
            qcut_ks_ = self.test_qcut_tb['KS'].max()
            self.test_auc_param = (y_, y_score_)
            auc_ = roc_auc_score(y_, y_score_)
            ap_ = average_precision_score(y_, y_score_)
            data['test'] = [auc_, ap_, uncut_ks_, cut_ks_, qcut_ks_]
            # 计算psi
            psi_cut = self._get_psi(self.train_cut_tb, self.test_cut_tb)
            psi_qcut = self._get_psi(self.train_qcut_tb, self.test_qcut_tb)
            data['psi'] = ['', '', '', psi_cut, psi_qcut]

        data['max_train_br'] = ['', '', '', self.train_cut_tb['default_percent'].max(), self.train_qcut_tb['default_percent'].max()]
        if self.enable_train_test:
            data['max_test_br'] = ['', '', '', self.test_cut_tb['default_percent'].max(), self.test_qcut_tb['default_percent'].max()]
        return data

    def get_psi_detail(self):
        data = pd.DataFrame(index=self.train_cut_tb.index)
        data['cut_psi'] = range(self.train_cut_tb.shape[0])
        data['train_scope'] = self.train_cut_tb.index.astype(str)
        data['train_total'] = self.train_cut_tb['total']
        data['train_percent'] = self.train_cut_tb['total_percent']
        data.index.name = 'cut'
        if self.enable_train_test:
            data['test_scope'] = self.test_cut_tb.index.astype(str)
            data['test_total'] = self.test_cut_tb['total']
            data['test_percent'] = self.test_cut_tb['total_percent']
            data['psi'] = data['train_percent'].div(data['test_percent']).map(log).mul(data["train_percent"].sub(data["test_percent"]))
        data2 = pd.DataFrame(index=self.train_qcut_tb.index)
        data2['cut_psi'] = range(self.train_qcut_tb.shape[0])
        data2['train_scope'] = self.train_qcut_tb.index.astype(str)
        data2['train_total'] = self.train_qcut_tb['total']
        data2['train_percent'] = self.train_qcut_tb['total_percent']
        data2.index.name = 'qcut'
        if self.enable_train_test:
            data2['test_scope'] = self.test_qcut_tb.index.astype(str)
            data2['test_total'] = self.test_qcut_tb['total']
            data2['test_percent'] = self.test_qcut_tb['total_percent']
            data2['psi'] = data2['train_percent'].div(data2['test_percent']).map(log).mul(data2["train_percent"].sub(data2["test_percent"]))
        return data, data2

    def get_model_basic(self, results):
        value = pd.DataFrame.from_dict(
            {'Dep. Variable': self.response,
             'No. Observations': results.nobs,
             'Df Residuals': results.df_resid,
             'Df Model': results.df_model,
             'Pseudo R-squ.': results.prsquared,
             'Log-Likelihood': results.llf,
             'LL-Null': results.llnull,
             'AIC': results.aic,
             'BIC': results.bic,
             'Cov_type': 'nonrobust'
             }, orient='index')  # results.cov_type
        value.columns = ['val']
        return value.sort_index()

    def get_model_summary(self, results):
        coef = results.params.to_frame()
        coef.columns = ['coef']
        std_err = results.bse.to_frame()
        std_err.columns = ['std_err']
        tvalues = results.tvalues.to_frame()
        tvalues.columns = ['tvalues']
        pvalues = results.pvalues.to_frame()
        pvalues.columns = ['pvalues']
        out_df = pd.concat([coef, std_err, tvalues, pvalues], axis=1)
        return out_df

    def get_score_group(self, model, cols):
        cols = list(set(cols) - {'Intercept'})
        cols = [col[4:] for col in cols]
        dfs = []
        for col in cols:
            tmp = self.df_i_d[self.df_i_d['var_name'] == col]
            dfs.append(tmp)

        coef_dict = model.params.to_dict()
        k = len(cols)
        B = self.score_basic['basic_b']
        A = self.score_basic['basic_a']
        dfs_ = []
        for df in dfs:
            tmp = df[['var_new_x', 'var_name', 'var_scope', 'WOE']]
            name = tmp['var_name'].values[0]
            key_name = "WOE_" + name
            tmp.loc[:, 'score'] = A / k - (tmp['WOE'].astype(float) * coef_dict[key_name] + coef_dict['Intercept'] / k) * B
            tmp.loc[:, 'score'] = tmp['score'].round(0)
            tmp.loc[:, name] = range(1, tmp.shape[0] + 1)
            tmp.set_index(name, inplace=True)
            dfs_.append(tmp)

        return dfs_

    def get_score(self, model, df, cols):
        cols = list(set(cols) - {'Intercept'})
        k = len(cols)
        data = df[cols]
        coef_dict = model.params.to_dict()
        # intercept = 1.0
        B = self.score_basic['basic_b']
        A = self.score_basic['basic_a']
        for name in data:
            data.loc[:, name] = A / k - (data[name] * coef_dict[name] + coef_dict['Intercept'] / k) * B
        data.loc[:, 'total_score'] = data.sum(axis=1)
        return data

    @staticmethod
    def get_var_corr(df, cols):
        cols = list(set(cols) - {'Intercept'})
        results = []
        for i in cols:
            for j in cols:
                results.append(pearsonr(df[i], df[j])[0])
        results = np.reshape(results, (len(cols), len(cols)))
        data = pd.DataFrame(results, index=cols, columns=cols)
        return data.round(5)

    def get_total_score(self, df, prob_score):
        data = pd.concat(prob_score, axis=1)
        if self.add_info is not None:
            data = pd.concat([data, df[self.add_info]], axis=1)
        return data

    def corr_filter(self, x, cols):
        show('Corr filter...')
        columns = self.df_i.sort_values(self.kwargs.get('select_by', 'iv'), ascending=False).index.tolist()
        columns = {"WOE_" + i for i in columns if "WOE_" + i in cols}
        data = []
        while len(columns) != 0:
            tmp_list = []
            left_col = columns.pop()
            for i in columns:
                p_ = pearsonr(x[left_col], x[i])[0]
                if p_ >= self.corr_limit:
                    tmp_list.append(i)
            data.append(left_col)
            columns -= set(tmp_list)
        return data

    def get_var_vif(self, df, cols):
        cols = list(set(cols) - {'Intercept'})
        features = " + ".join(cols)
        use_df = df[cols + [self.response]]
        y, x = dmatrices(self.response + ' ~ ' + features, use_df, return_type='dataframe')
        data = pd.DataFrame()
        data.loc[:, 'var'] = x.columns
        data.loc[:, 'ols_vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        return data

    def get_report(self):
        r_excel_name = 'R_sm_' + self.filename + '.xlsx'

        woe_col = [i for i in self.df_m if i.startswith("WOE")]
        x = self.df_m[woe_col]
        if self.exclude_column is not None:
            x = x.drop(self.exclude_column, axis=1)
        y = self.df_m[self.response]
        cols = x.columns

        if self.kwargs.get('enable_corr_filter', True) is True:  # 相关性筛选
            cols = self.corr_filter(x, cols)

        if self.kwargs.get('filter_var', True) is True:  # 基于模型的变量筛选
            x, y, cols = self.model_filter.get(self.kwargs.get('model_type', 'sm'))(x, y)

        writer = pd.ExcelWriter(r_excel_name, engine='xlsxwriter')
        model = self.get_standard_model(x, y, cols)
        results_summary = self.get_results_summary(model, x, cols)
        psi_detail1, psi_detail2 = self.get_psi_detail()

        model_basic = self.get_model_basic(model)
        self.append_to_excel(model_basic, writer, 'model_basic')

        model_summary = self.get_model_summary(model)
        self.append_to_excel(model_summary, writer, 'model_summary')

        var_summary = self.get_var_summary(cols)
        self.append_to_excel(var_summary, writer, 'var_summary')

        var_detail = self.get_var_detail(cols)
        self.append_to_excel(var_detail, writer, 'var_detail', index=False)

        train_var_corr = self.get_var_corr(self.df_m, cols)
        self.append_to_excel(train_var_corr, writer, 'train_var_corr')

        train_var_vif = self.get_var_vif(self.df_m, cols)
        self.append_to_excel(train_var_vif, writer, 'train_var_vif')

        train_score = self.get_score(model, self.df_m, cols)
        self.append_to_excel(train_score, writer, 'train_score')

        score_group = self.get_score_group(model, cols)
        self.append_to_excel(score_group, writer, 'score_group')

        score_basic = self.get_score_basic()
        self.append_to_excel(score_basic, writer, 'score_basic')

        if self.save_proba is True:
            train_total_score = self.get_total_score(self.df_m, self.train_proba_score)
            self.append_to_excel(train_total_score, writer, sheet_name='train_total_score')
            if self.enable_train_test is True:
                test_total_score = self.get_total_score(self.df_t, self.test_proba_score)
                self.append_to_excel(test_total_score, writer, sheet_name='test_total_score')

        Plot.plot_ap_curve(self.train_auc_param, writer, 'train_ap_curve')
        Plot.plot_roc_curve(self.train_auc_param, writer, 'train_roc_curve')

        self.append_to_excel(self.train_cut_tb, writer, 'train_score_cut')
        Plot.plot_cut_graph(self.train_cut_tb, writer, 'train_score_cut_graph')

        self.append_to_excel(self.train_qcut_tb, writer, 'train_score_qcut')
        Plot.plot_cut_graph(self.train_qcut_tb, writer, 'train_score_qcut_graph')
        if self.enable_train_test is True:
            test_score = self.get_score(model, self.df_t, cols)
            self.append_to_excel(test_score, writer, 'test_score')

            test_var_corr = self.get_var_corr(self.df_t, cols)
            self.append_to_excel(test_var_corr, writer, "test_var_corr")

            test_var_vif = self.get_var_vif(self.df_t, cols)
            self.append_to_excel(test_var_vif, writer, "test_var_vif")

            Plot.plot_ap_curve(self.test_auc_param, writer, 'test_ap_curve')
            Plot.plot_roc_curve(self.test_auc_param, writer, 'test_roc_curve')

            self.append_to_excel(self.test_cut_tb, writer, 'test_score_cut')
            Plot.plot_cut_graph(self.test_cut_tb, writer, 'test_score_cut_graph')

            self.append_to_excel(self.test_qcut_tb, writer, 'test_score_qcut')
            Plot.plot_cut_graph(self.test_qcut_tb, writer, 'test_score_qcut_graph')

        self.append_to_excel([psi_detail1, psi_detail2], writer, 'psi_detail')
        self.append_to_excel(results_summary, writer, 'results_summary')
        writer.save()
        Plot.rm_tmp_file()  # 删除临时文件
        show("文件已经保存到:{0}".format(r_excel_name))


class Plot(object):
    tmp_file = 'mc_tmp_files'

    @staticmethod
    def mkdir_tmp_file():
        if not os.path.exists(Plot.tmp_file):
            os.mkdir(Plot.tmp_file)

    @staticmethod
    def rm_tmp_file():
        show('Remove tmp file:{0}'.format(Plot.tmp_file))
        shutil.rmtree(Plot.tmp_file)

    @staticmethod
    def append_img_excel(imgs, writer, sheet_name):
        show('Add to excel:{0}'.format(sheet_name))
        book = writer.book
        sheet = book.add_worksheet(sheet_name)
        if isinstance(imgs, (list, tuple, set)):
            startrow = 1
            for img in imgs:
                sheet.insert_image('A{0}'.format(startrow), img)
                startrow += 30
        else:
            sheet.insert_image('A1', imgs)

    @staticmethod
    def save_ks_graph(df, sheet_name):
        Plot.mkdir_tmp_file()
        filename = os.path.join(Plot.tmp_file, sheet_name + '_ks.png')
        df = copy.deepcopy(df)
        df['x'] = range(df.shape[0])
        x = df['x'].values
        plt.figure(figsize=(10, 6), dpi=60, facecolor='white')
        plt.plot(x, df['cum_good_percent'], linestyle='solid', linewidth=1, color='blue', label='Cumnlative Good Percent')
        plt.plot(x, df['cum_bad_percent'], linestyle='solid', linewidth=1, color='red', label='Cumnlative Bad Percent')
        ks = df['KS'].max()
        ks_y_max = df.loc[df.idxmax()['KS'], :]['cum_bad_percent']
        ks_y_min = df.loc[df.idxmax()['KS'], :]['cum_good_percent']
        ks_x = df.loc[df.idxmax()['KS'], :]['x']
        plt.plot([ks_x, ks_x], [ks_y_max, ks_y_min], linestyle='--', linewidth=3, color='black', label='Kolmogorov-Smirnov(KS):{0}'.format(ks))
        plt.legend(loc='lower right')
        plt.ylim(0, 1)
        ax = plt.gca()
        x_labels = df.index.values
        ax.set_xticklabels(x_labels, rotation=40, fontsize=8, color='b')
        plt.title('Equivalent-based Kolmogorov-Smirnov(KS) Graph')
        plt.gcf().savefig(filename, dpi=100)
        return filename

    @staticmethod
    def save_df_graph(df, sheet_name):
        Plot.mkdir_tmp_file()
        filename = os.path.join(Plot.tmp_file, sheet_name + '_df.png')
        df = copy.deepcopy(df)
        df['x'] = range(df.shape[0])
        x = df['x'].values
        plt.figure(figsize=(10, 6), dpi=60, facecolor='white')
        plt.plot(x, df['default_percent'], linestyle='solid', linewidth=1, color='blue', label='Default Rate')
        plt.legend(loc='upper right')
        for x, y in zip(df['x'], df['default_percent']):
            plt.text(x, y, '{0:.2f}%'.format(y * 100), ha='center', fontsize=10)
        ax = plt.gca()
        x_labels = df.index.values
        ax.set_xticklabels(x_labels, rotation=40, fontsize=9, color='b')
        plt.title('Equivalent-based Default Rate Curve')
        plt.gcf().savefig(filename, dpi=100)
        return filename

    @staticmethod
    def plot_roc_curve(params, writer, sheet_name):
        Plot.mkdir_tmp_file()
        filename = os.path.join(Plot.tmp_file, sheet_name + '.png')
        false_positive_rate, true_positive_rate, thresholds = roc_curve(params[0], params[1])
        roc_auc = roc_auc_score(params[0], params[1])
        fig, ax = plt.subplots(1)
        ax.set_title("Receiver Operating Characteristic(ROC)")
        ax.plot(false_positive_rate, true_positive_rate, c='#2B94E9', label='AUC = %0.3f' % roc_auc)
        ax.legend(loc='lower right')
        ax.plot([0, 1], [0, 1], 'm--', c='#666666')
        plt.xlim([0, 1])
        plt.ylim([0, 1.1])
        plt.gcf().savefig(filename, dpi=100)
        plt.close('all')
        Plot.append_img_excel(filename, writer, sheet_name)

    @staticmethod
    def plot_cut_graph(df, writer, sheet_name):
        ks_img = Plot.save_ks_graph(df, sheet_name)
        df_img = Plot.save_df_graph(df, sheet_name)
        plt.close('all')
        Plot.append_img_excel([ks_img, df_img], writer, sheet_name)

    @staticmethod
    def plot_ap_curve(params, writer, sheet_name):
        Plot.mkdir_tmp_file()
        filename = os.path.join(Plot.tmp_file, sheet_name + '.png')
        fig, ax = plt.subplots(1)
        ap = average_precision_score(params[0], params[1])
        p, r, _ = precision_recall_curve(params[0], params[1])
        ax.plot(r, p, c='#2B94E9', label='Average Precision(AP):%0.2f' % ap)
        ax.legend(loc='upper right')
        ax.set_title("Precision Recall CURVE(PR)")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([-0.05, 1.1])
        plt.ylim([-0.05, 1.1])
        plt.gcf().savefig(filename, dpi=100)
        plt.close('all')
        Plot.append_img_excel(filename, writer, sheet_name)


if __name__ == '__main__':
    mc = ModelControl('mg1121', enable_train_test=True, response="code", add_info=['code', 'umobile', 'uid', 'date'], save_proba=True)
    mc.get_report()
