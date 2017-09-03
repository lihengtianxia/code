# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
from __future__ import unicode_literals, division
import json
import pandas as pd
import os
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
filepath = os.path.split(os.path.realpath(__file__))[0]


# def numeric_binning(s, val):
#     if (s == u"-999" or s == -999) and val == -999:
#         return True
#     if (s == u"-1" or s == -1) and val == -1:
#         return True
#     if (s == u"-1" or s == -1) and val != -1:
#         return False
#     s = s.strip(" ")
#     _ele = []
#     _ele.append(s[0])  # ["["]
#     _ele.append(s[len(s) - 1])  # ["[","]"]
#     flag = 0
#     val = round(val, 3)
#     if s.find('+') >= 0:
#         _ele.append(float(s[1:len(s) - 1].replace(' ', '').split(",")[0]))
#         if _ele[0] == "(" and _ele[2] < val:
#             flag += 2
#         elif _ele[0] == "[" and _ele[2] <= val:
#             flag += 2
#     elif s.find('+') < 0:
#         _ele.extend([float(x) for x in s[1:len(s) - 1].replace(' ', '').split(",")])
#         if _ele[0] == "(" and _ele[2] < val:
#             flag += 1
#         elif _ele[0] == "[" and _ele[2] <= val:
#             flag += 1
#         if _ele[1] == ")" and _ele[3] > val:
#             flag += 1
#         elif _ele[1] == "]" and _ele[3] >= val:
#             flag += 1
#     if flag == 2:
#         return True
#     else:
#         return False
#
#
# def categorical_binning(s, val):
#     # s = s.strip(" ")
#     s = s.strip().split("|")
#     if val in s:
#         return True
#     # elif val == s:
#     #     return True
#     else:
#         return False
#

# def binning(df_, val):
#     import copy
#     df = copy.copy(df_)
#     var_name = df.iloc[0].tolist()
#     df.index = [x for x in range(df.shape[0])]
#     df = df.drop(0)
#     df.columns = var_name
#     label_ = -1
#     range_ = ''
#     score_ = 0
#
#     # if type(val) == type('') or type(val) == type(u''):
#     if isinstance(val, (str, unicode, bytes)):
#         # 对于有数字的分类信息，如渤海的y_latest_in_amt_min是否有可能有误?
#         # 判断在哪个bin
#         for i in range(len(df)):
#             # if categorical_binning(df.iloc[i][1], val) == True:
#             if categorical_binning(df.iloc[i][1], val):
#                 # 输出label，range和score
#                 label_ = df.iloc[i][0]
#                 range_ = df.iloc[i][1]
#                 score_ = df.iloc[i][2]
#                 return label_, range_, score_
#     # elif type(val) == type(1.0) or type(val) == type(1):
#     elif isinstance(val, (float, int)):
#         for i in range(len(df)):
#             # if numeric_binning(df.iloc[i][1], val) == True:
#             if numeric_binning(df.iloc[i][1], val):
#                 # 输出label，range和score
#                 label_ = df.iloc[i][0]
#                 range_ = df.iloc[i][1]
#                 score_ = df.iloc[i][2]
#                 return label_, range_, score_
#     return label_, range_, score_
def judge_sdf_and_value(value, sdf):
    if isinstance(value, (str, unicode, bytes)):
        # if type(value) in [str, unicode]:
        sdf = sdf.strip(" ")
        if value in sdf:
            return True
        elif value == sdf:
            return True
        else:
            return False
    if isinstance(value, (float, int)):
        # if type(value) in [float, int]:
        value = float(value)
        try:
            sdf = float(sdf)
            return value == sdf
        except ValueError:
            front_boundary = sdf[0]
            end_boundary = sdf[-1]
            sdf_arr = sdf[1: len(sdf) - 1].split(',')
            if (front_boundary == '(' and value > float(sdf_arr[0])) or \
                    (front_boundary == '[' and value >= float(sdf_arr[0])):
                if sdf_arr[1].strip() == '+':
                    return True
                else:
                    if (end_boundary == ')' and value < float(sdf_arr[1])) \
                            or (end_boundary == ']' and value <= float(sdf_arr[1])):
                        return True

    return False


def binning(df_, val):
    import copy
    df = copy.copy(df_)
    var_name = df.iloc[0].tolist()
    df.index = [x for x in range(df.shape[0])]
    df = df.drop(0)
    df.columns = var_name
    for i in range(len(df)):
        sdf = df.iloc[i][1]
        if judge_sdf_and_value(val, sdf):
            return df.iloc[i][0], df.iloc[i][1], df.iloc[i][2]

    return -1, '', 0


def handle(params):
    import math
    df_all = pd.read_excel(os.path.join(filepath + "/bohai_bin.xlsx"), header=None)
    result = {"credit_score": 0}
    df_all[3] = df_all[3].fillna("NaN").map(lambda _: _.upper())
    # df_all[3] = [x.upper() if type(x) == type(u'') else x for x in df_all[3].tolist()]
    for key in params.keys():
        # feidai_bin第三列作为分
        if key.upper() in df_all[3].tolist():
            df_ = df_all[df_all[3] == key.upper()]
            val = params.get(key)
            # print key + ": " + str(val)
            # if val == '' or val == u'' or val == u'{}' or val == '{}':
            if val == '' or val == '{}':
                val = '-999'
            try:
                label, range, score = binning(df_, val)
                print key + "_score: " + str(score)
            except Exception, e:
                print e
                print '变量' + str(key) + "有问题, " + str(df_[1].tolist())
            if key != 'economic_stability_level_12' and key != 'y_latest_in_amt_min':
                # 信用分加和
                # result["credit_score"] = score + result["credit_score"]
                result["credit_score"] += score
            # 特殊字段名字处理
            if key == 'third_part_degree':
                result['ext_diploma_score'] = score
            elif key == 'economic_stability_level_12':
                result['ext_economic_stability'] = score
            elif key == 'y_latest_in_amt_min':
                result['ext_pay_year'] = score
            else:
                result["ext_" + key.lower().replace('_info', '').replace('psg_', '') + '_score'] = score
                # result['l_'+key.lower()]=label
        else:
            raise ValueError("{0} not in dataframe!".format(key))
    df_score = pd.read_excel(filepath + "/bohai_bin.xlsx", header=None, sheetname='评分等级')
    label_band, range_band, score_band = binning(df_score, result['credit_score'])
    result['credit_limit'] = 0
    result['interest_rate_grade'] = ''
    result['credit_score'] = round(result['credit_score'], 2)
    # result['credit_decision'] = 'Review'
    if score_band == 'E' or score_band == 'D' or (score_band == 'C' and result['ext_economic_stability'] == '特殊值未查得'):
        result['credit_decision'] = 'Reject'
    else:
        if (score_band == 'A' and result['ext_economic_stability'] == '稳定') or (score_band == 'A' and result['ext_economic_stability'] == '准稳定') or (
                        score_band == 'A' and result['ext_economic_stability'] == '不稳定') or (score_band == 'B' and result['ext_economic_stability'] == '稳定'):
            result['credit_decision'] = 'Accept'
            result['interest_rate_grade'] = 'A'
            if result['ext_pay_year'] != '未查得':
                result['credit_limit'] = result['ext_pay_year'] * 0.25
            else:
                result['credit_limit'] = 4000
        else:
            if (score_band == 'A' and result['ext_economic_stability'] == '很不稳定') or (score_band == 'A' and result['ext_economic_stability'] == '极不稳定') or (
                            score_band == 'A' and result['ext_economic_stability'] == '特殊值未查得') or (score_band == 'B' and result['ext_economic_stability'] == '准稳定') or (
                            score_band == 'B' and result['ext_economic_stability'] == '不稳定'):
                result['credit_decision'] = 'Accept'
                result['interest_rate_grade'] = 'B'
                if result['ext_pay_year'] != '未查得':
                    result['credit_limit'] = result['ext_pay_year'] * 0.2
                else:
                    result['credit_limit'] = 3000
            else:
                if (score_band == 'B' and result['ext_economic_stability'] == '很不稳定') or (score_band == 'B' and result['ext_economic_stability'] == '极不稳定') or (
                                score_band == 'B' and result['ext_economic_stability'] == '特殊值未查得') or (score_band == 'C' and result['ext_economic_stability'] == '稳定'):
                    result['credit_decision'] = 'Accept'
                    result['interest_rate_grade'] = 'C'
                    if result['ext_pay_year'] != '未查得':
                        result['credit_limit'] = result['ext_pay_year'] * 0.15
                    else:
                        result['credit_limit'] = 2000
                else:
                    if (score_band == 'C' and result['ext_economic_stability'] == '很不稳定') or (score_band == 'C' and result['ext_economic_stability'] == '极不稳定') or (
                                    score_band == 'C' and result['ext_economic_stability'] == '不稳定') or (score_band == 'C' and result['ext_economic_stability'] == '准稳定'):
                        result['credit_decision'] = 'Accept'
                        result['interest_rate_grade'] = 'D'
                        if result['ext_pay_year'] != '未查得':
                            result['credit_limit'] = result['ext_pay_year'] * 0.1
                        else:
                            result['credit_limit'] = 1000
    result['credit_score'] = int(round(result['credit_score']))
    if result['credit_decision'] == 'Accept' and result['credit_limit'] < 1000 and result['credit_limit'] >= 0:
        result['credit_limit'] = 1000
    if result['credit_decision'] == 'Accept' and result['credit_limit'] > 30000:
        result['credit_limit'] = 30000
    return result
