# coding:utf-8
# 通用的计算IV和WOE代码，注意控制变量的水平个数，最好不要超过100个，不然计算会报错。
计算IV_WOE

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from operator import add
 
def attr_cnt(row, attr_list):


    '''各attr变量下不同value数量统计 for flatMap
       Args:
          row:行
          attr_val_dict:{attr:value_set}
       Return:
          由((attr,value),1) 组成的list
       '''
   row_dict = row.asDict()
   result = []
   for attr in attr_list:
          result.append(((attr, row_dict.get(attr, None)), 1))
   return result
 
def cal_woe(attr, attr_val_cnt_all, attr_val_cnt_good, none_avalible=True):

       '''计算某一attr的woe,IV
   Args:
      attr_val_dict:各attr变量的distinct values字典
      attr:特定的attr变量
      attr_val_cnt_all:以各attr,value为index，all_count为column的DataFrame
      attr_val_cnt_good:以各attr,value为index，good_count为column的DataFrame
      none_avalible: boolean，是否把None值纳入计算范围
   Return:
      特定attr变量的iv、woe计算结果(pandas DataFrame)
   '''
   attr_values = list(attr_val_cnt_all.loc[attr].index.values)
   attr_count = {}
   attr_good_count = {}
   attr_bad_count = {}
   all_dist = {}
   good_dist = {}
   bad_dist = {}
   if none_avalible:
          total_num = sum(attr_val_cnt_all.loc[attr]['all_count'])
      total_num_good = sum(attr_val_cnt_good.loc[attr]['good_count'])
      total_num_bad = total_num - total_num_good
   else:
      total_num = sum(attr_val_cnt_all.loc[attr].loc[pd.notnull(attr_val_cnt_all.loc[attr].index)]['all_count'])
      total_num_good = sum(attr_val_cnt_good.loc[attr].loc[pd.notnull(attr_val_cnt_good.loc[attr].index)]['good_count'])
      total_num_bad = total_num - total_num_good
   for val in attr_values:
          if pd.notnull(val):
             attr_count[val] = attr_val_cnt_all.loc[attr, val]['all_count']  # attr特定类型数量
         try:
                attr_good_count[val] = attr_val_cnt_good.loc[attr, val]['good_count']  # attr特定类型good数量
         except:
            attr_good_count[val] = 0
         attr_bad_count[val] = attr_count[val] - attr_good_count[val]  # attr特定类型bad数量
         all_dist[val] = attr_count[val] / float(total_num)  # attr样本百分比
         good_dist[val] = attr_good_count[val] / float(total_num_good)  # good to good all
         bad_dist[val] = attr_bad_count[val] / float(total_num_bad)  # bad to bad all
      else:
         if none_avalible:
                all_count = attr_val_cnt_all.loc[attr].loc[pd.isnull(attr_val_cnt_all.loc[attr].index)].iloc[
    0, 0]  # attr特定类型数量
            good_count = attr_val_cnt_good.loc[attr].loc[pd.isnull(attr_val_cnt_good.loc[attr].index)].iloc[
    0, 0]  # attr特定类型good数量
            bad_count = all_count - good_count  # attr特定类型bad数量
            attr_count[val] = all_count
            attr_good_count[val] = good_count
            attr_bad_count[val] = bad_count
            all_dist[val] = all_count / float(total_num)  # attr样本百分比
            good_dist[val] = good_count / float(total_num_good)  # good to good all
            bad_dist[val] = bad_count / float(total_num_bad)  # bad to bad all
         else:
            pass
   attr_count = pd.DataFrame(attr_count.items(), columns=[attr, '样本数量']).set_index(attr)
   attr_good_count = pd.DataFrame(attr_good_count.items(), columns=[attr, '好客户数量']).set_index(attr)
   attr_bad_count = pd.DataFrame(attr_bad_count.items(), columns=[attr, '坏客户数量']).set_index(attr)
   all_dist = pd.DataFrame(all_dist.items(), columns=[attr, '样本占比']).set_index(attr)
   good_dist = pd.DataFrame(good_dist.items(), columns=[attr, '好客户占比']).set_index(attr)
   bad_dist = pd.DataFrame(bad_dist.items(), columns=[attr, '坏客户占比']).set_index(attr)
   woe = pd.concat([attr_count, attr_good_count, attr_bad_count, all_dist, good_dist, bad_dist], axis=1)
   try:  # good_dist或bad_dist有一个为0则woe break了
          woe['WOE'] = np.log(woe['好客户占比'] / woe['坏客户占比'])
      woe['IV'] = np.round((woe['好客户占比'] - woe['坏客户占比']) * woe['WOE'], 5)
      woe = woe.sort_index(na_position='first')
      total_value = pd.DataFrame([[sum(woe['样本数量']), sum(woe['好客户数量']), sum(woe['坏客户数量']), sum(woe['样本占比']),
                                   sum(woe['好客户占比']), sum(woe['坏客户占比']), sum(woe['WOE']), sum(woe['IV'])]],
                                 index=['合计'],
                                 columns=['样本数量', '好客户数量', '坏客户数量', '样本占比', '好客户占比', '坏客户占比', 'WOE', 'IV'])
      woe = woe.append(total_value)
   except:
      pass
   return woe
 
def dfs2xlsx(list_dfs, xls_path=None):

        '''将df列表写入excel同一sheet下
    Args:
        list_dfs:dataframe列表
        xls_path:写出的文件路径
    '''
    if xls_path == None:
            xls_path = '~tmp.xlsx'
    writer = ExcelWriter(xls_path)
    i = 0
    for df in list_dfs:
            df.to_excel(writer, 'Sheet1', startrow=i, encoding="utf8")
        i += len(df) + 2
    writer.save()
 
 
def calc_iv(data, col):

        import pandas as pd
    attr_list = [col]
    rule_hit_cnt = data.flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(add).collect()  # 各attr变量不同value数量统计
    rule_hit_good = data.filter('fraudFlag = 0').flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(
    add).collect()  # 各attr变量不同value下target变量为good数量统计
 
      # 转成以各attr,value为index，count为column的DataFrame，方便计算woe、iv
    rule_hit_cnt = pd.DataFrame([(attr, val, count) for ((attr, val), count) in rule_hit_cnt],
                                columns=['attr', 'value', 'all_count']).set_index(['attr', 'value'])
    rule_hit_good = pd.DataFrame([(attr, val, count) for ((attr, val), count) in rule_hit_good],
                                 columns=['attr', 'value', 'good_count']).set_index(['attr', 'value'])
    df_woe = cal_woe(col, rule_hit_cnt, rule_hit_good)
 
    return df_woe
 
data = sqlContext.read.json("/user/credit/rule_hit1029")
attr_list = ['rule%d' % i for i in range(1, 99)]
 
rule_hit_cnt = data.flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(add).collect()  # 各attr变量不同value数量统计
rule_hit_good = data.filter('fraud_flag = 0').flatMap(lambda x: attr_cnt(x, attr_list)).reduceByKey(
    add).collect()  # 各attr变量不同value下target变量为good数量统计
 
# 转成以各attr,value为index，count为column的DataFrame，方便计算woe、iv
rule_hit_cnt = pd.DataFrame([(attr, val, count) for ((attr, val), count) in rule_hit_cnt],
                            columns=['attr', 'value', 'all_count']).set_index(['attr', 'value'])
rule_hit_good = pd.DataFrame([(attr, val, count) for ((attr, val), count) in rule_hit_good],
                             columns=['attr', 'value', 'good_count']).set_index(['attr', 'value'])
 
iv_woes = []
for attr in attr_list:
       df = cal_woe(attr, rule_hit_cnt, rule_hit_good)
   iv_woes.append(df)
 
dfs2xlsx(iv_woes, '/home/chao.qin/suning_iv2.xlsx')
