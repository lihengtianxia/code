#分bin计算woe和IV 3.0版本，添加对分类变量的计算，使用排序后的最大woe之差，以及ks。
__author__ = 'fraudmetrix-chaoqin'
# !/usr/bin/env python
# -*- coding:utf-8 -*-
 
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from operator import add
 
# 根据输入的dict得到特定dict的key的排序，返回key和对应的最大值位置
def get_max_place(ks_dict, key_place):
    ks_keys = ks_dict.keys()
    ks_keys.sort(key=lambda x: x[key_place])
    ks_rank = [i[key_place] for i in ks_keys]
    max_ks = ks_rank[ks_rank.index(max(ks_rank))]
    max_place = [i[key_place] == max_ks for i in ks_keys]
    return max_place, ks_keys
 
 
# 默认读取df_的index内容进行分组添加，同时添加追踪标记
def add_info(t_index, df_, info, out):
    info = info.drop('cum_good_cnt', axis=1).drop('cum_bad_cnt', axis=1).drop(
        'cum_good_pct', axis=1).drop('cum_bad_pct', axis=1)
    info.ix[:, df_.index.name] = [
        ', '.join(df_.ix[(df_[out] == i)].index.tolist()) for i in info.index]
    info.ix[:, 'index'] = [t_index for _ in range(len(info))]
    info = info.set_index(df_.index.name)
    cols = info.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    info = info[cols]
    return info
 
 
# 根据输入的df_分组，计算并返回ks，std和分组后的表
def cal_ks_std(df_, out, rank, good, bad, stability):
    info = df_.groupby(out)[[rank, stability, good, bad]].sum()
 
    if rank is 'WOE':
        good_sum = [float(info[good].sum()) for _ in range(len(info[good]))]
        bad_sum = [float(info[bad].sum()) for _ in range(len(info[bad]))]
        good_percent = np.array([float(i)
                                 for i in info[good]]) / np.array(good_sum)
        bad_percent = np.array([float(i)
                                for i in info[bad]] / np.array(bad_sum))
        info = info.drop(rank, axis=1)
        info.ix[:, rank] = np.log(good_percent / bad_percent)
        info.ix[:, 'IV'] = np.round(
            ((good_percent - bad_percent) * info[rank]), 5)
    else:
        info.ix[:, rank] = df_.groupby(out)[rank].mean()
        info.ix[:, 'IV'] = np.round(
            ((good_percent - bad_percent) * info['WOE']), 5)
 
    info = info.sort_values(rank)
    cum_good_cnt, cum_bad_cnt = [], []
    for i in range(len(info[good])):
        cum_good_cnt.append(info[good][:i + 1].sum())
    for i in range(len(info[bad])):
        cum_bad_cnt.append(info[bad][:i + 1].sum())
    info.ix[:, 'cum_good_cnt'] = cum_good_cnt
    info.ix[:, 'cum_bad_cnt'] = cum_bad_cnt
    info.ix[:, 'cum_good_pct'] = np.array([float(i) for i in cum_good_cnt]) / np.array(
        [float(cum_good_cnt[-1]) for _ in range(len(cum_good_cnt))])
    info.ix[:, 'cum_bad_pct'] = np.array([float(i) for i in cum_bad_cnt]) / np.array(
        [float(cum_bad_cnt[-1]) for _ in range(len(cum_bad_cnt))])
    info.ix[:, 'ks'] = abs(info['cum_good_pct'] - info['cum_bad_pct'])
    ks_ = info['ks'].max()
    std = info[stability].std()
    return ks_, std, info
 
 
def clear_inf(df_, value_):
    if np.inf in df_[value_].tolist():
        max_value = list(set(df_[value_].tolist()))
        max_value.sort()
        df_.ix[:, value_] = [(max_value[-2] + 0.001 if i == np.inf else i)
                             for i in df_[value_].tolist()]
    if -np.inf in df_[value_].tolist():
        min_value = list(set(df_[value_].tolist()))
        min_value.sort()
        df_.ix[:, value_] = [(min_value[1] - 0.001 if i == -np.inf else i)
                             for i in df_[value_].tolist()]
    return df_
 
 
# 根据WOE使用分位数分割，作为备选
def get_group_by_qcut(df_, part, out, value_):
    df_ = df_.sort_values(value_)
    df_ = clear_inf(df_, value_)
 
    group = pd.qcut(df_[value_], part)
 
    df_.ix[:, out] = group
 
    group_dict = {}
    for num, element in enumerate(list(set(group))):
        group_dict[element] = out + str(num + 1)
 
    df_.ix[:, out] = [group_dict[i] for i in group]
    return df_
 
 
# 根据WOE排序选取最大间距，遍历后根据ks和std选取特定分组最优解
def get_group_by_cal(df_, part, out, max_part, value_, good, bad, stability):
    df_ = df_.sort_values(value_)
    df_ = clear_inf(df_, value_)
 
    temp1, temp2 = np.array(df_[value_][1:]), np.array(df_[value_][:-1])
    woe_change = (temp1 - temp2).tolist()
    woe_change.append(0)
 
    place = []
    for _ in range(len(woe_change)):
        max_place = woe_change.index(max(woe_change))
        place.append(max_place + 1)
        woe_change[max_place] = 0
 
    if max_part < 10:
        place = place[:10]
    else:
        place = place[:max_part]
 
    ks_dict, std_dict = {}, {}
    for comb in combinations(place, part):
        comb = list(comb)
        comb.sort(reverse=True)
 
        woe_change[:] = [out + str(len(comb) + 1)
                         for _ in range(len(woe_change))]
        for ind_, val_ in enumerate(comb):
            woe_change[:val_] = [out + str(len(comb) - ind_)
                                 for _ in range(val_)]
        df_.ix[:, out] = woe_change
        ks_, std, info = cal_ks_std(df_, out, value_, good, bad, stability)
        std_dict[std] = comb
        ks_dict[(part, ks_, std)] = (info, df_)
 
    max_place, ks_keys = get_max_place(ks_dict, 1)
 
    ks_rank = []
    for status, content in izip_longest(max_place, ks_keys):
        if status:
            ks_rank.append(content)
    ks_rank.sort(key=lambda x: x[2])
 
    return ks_rank[0], ks_dict[ks_rank[0]]
 
 
# 从0到最大分组，分别使用qcut分组，获取每一分组的最优解
def output_by_qcut(t_index, df_, out, value_, good, bad, stability, max_part, output_list, output_folder):
    ks_dict, error_sum = {}, []
    for i in range(max_part):
        try:
            df_ = get_group_by_qcut(df_, i, out, value_)
            ks_, std, info = cal_ks_std(df_, out, value_, good, bad, stability)
            ks_dict[(i, ks_, std)] = (info, df_)
        except (ValueError, IndexError) as error:
            error_sum.append(error)
 
    max_place, rank_list = get_max_place(ks_dict, 1)
 
    for status, content in izip_longest(max_place, rank_list):
        if status:
            output_list.append(
                add_info(t_index, ks_dict[content][1], ks_dict[content][0], out))
            t_index += 1
    return t_index
 
 
# 从0到最大分组，分别使用最大间隔计算分组，获取每一分组的最优解
def output_by_cal(t_index, df_, out, value_, good, bad, stability, max_part, output_list, output_folder):
    ks_dict, error_sum = {}, []
    for i in range(max_part):
        try:
            out_key, out_value = get_group_by_cal(
                df_, i, out, max_part, value_, good, bad, stability)
            ks_dict[out_key] = out_value
        except (ValueError, IndexError) as error:
            error_sum.append(error)
 
    max_place, rank_list = get_max_place(ks_dict, 1)
 
    for status, content in izip_longest(max_place, rank_list):
        if status:
            output_list.append(
                add_info(t_index, ks_dict[content][1], ks_dict[content][0], out))
            t_index += 1
    return t_index
 
 
# 获取最优分组
def get_optimal_group(df_, t_index=0, value_='WOE', out='分组', good='好客户数量', bad='坏客户数量', stability='样本数量', max_part=10, output=False, output_folder='output'):
    i, opath, output_list = 0, os.getcwd(), []
    if not os.path.exists(os.path.join(os.getcwd(), output_folder)):
        os.mkdir(os.path.join(os.getcwd(), output_folder))
    if '合计' in df_.index.tolist():
        df_ = df_.ix[(df_.index != '合计')]
    # t_index = output_by_qcut(t_index, df_, out, value_, good,
    # bad, stability, max_part, output_list, output_folder)
    t_index = output_by_cal(t_index, df_, out, value_, good, bad, stability,
                            max_part, output_list, output_folder)
    if output:
        os.chdir(os.path.join(os.getcwd(), output_folder))
        excel_file = pd.ExcelWriter(
            output_list[0].index.name + '_group_reference.xlsx')
        for df_ in output_list:
            df_.to_excel(excel_file, 'Sheet1', startrow=i)
            i += len(df_) + 2
        excel_file.save()
        os.chdir(opath)
    return output_list, t_index
 
 
# 为原始数据添加最优分组列和对应的WOE列
def add_group_to_df(df_pd, df_add):
    target = [i for i in df_pd.columns.tolist() if i == df_add.index.name]
    part_dict, woe_dict = {}, {}
    d_type = df_add.index.values.tolist()[0]
    if isinstance(d_type, str) or isinstance(d_type, unicode):
        for group in df_add.index.values.tolist():
            for part in group.split(', '):
                part_dict[part] = group
                woe_dict[part] = df_add.ix[(df_add.index == group)][
                    'WOE'].tolist()[0]
        df_pd.ix[:, target[0] + '_Q'] = [part_dict[element]
                                         for element in df_pd[target[0]].tolist()]
        df_pd.ix[:, target[0] + '_WOE'] = [woe_dict[element]
                                           for element in df_pd[target[0]].tolist()]
    else:
        ori = df_add.index.values.tolist()
        check = [float(i) for i in ori[:-1]]
        value_ = [float(i) for i in df_pd[target[0]].tolist()]
        for info in value_:
            status = [ori[i] <= info < ori[i + 1]
                      for i in range(len(check))]
            if True in status:
                part_dict[info] = ori[status.index(True)]
                woe_dict[info] = df_add.ix[
                    (df_add.index == ori[status.index(True)])]['WOE'].tolist()[0]
            else:
                part_dict[info] = ori[-1]
                woe_dict[info] = df_add.ix[
                    (df_add.index == ori[-1])]['WOE'].tolist()[0]
        df_pd.ix[:, target[0] + '_Q'] = [part_dict[float(element)]
                                         for element in df_pd[target[0]].tolist()]
        df_pd.ix[:, target[0] + '_WOE'] = [woe_dict[float(element)]
                                           for element in df_pd[target[0]].tolist()]
    return df_pd
 
 
 
def attr_cnt(df_pd, col_rename):
   '''各attr变量下不同value数量统计 for flatMap
   Args:
      row:行
      attr_val_dict:{attr:value_set}
   Return:
      由((attr,value),1) 组成的list
   '''
   result=[]
   group_data=df_pd.groupby(col_rename).size()
   group_table=group_data.reset_index()
   for i in range(len(group_table)):
      result.append(((col_rename, group_table[col_rename][i]), group_table[0][i]))
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
         attr_count[val] = attr_val_cnt_all.loc[attr,val]['all_count'] # attr特定类型数量
         try:
            attr_good_count[val] = attr_val_cnt_good.loc[attr,val]['good_count'] # attr特定类型good数量
         except:
            attr_good_count[val] = 0
         attr_bad_count[val] = attr_count[val] - attr_good_count[val] # attr特定类型bad数量
         all_dist[val] = attr_count[val]/float(total_num) # attr样本百分比
         good_dist[val] = attr_good_count[val]/float(total_num_good) # good to good all
         bad_dist[val] = attr_bad_count[val]/float(total_num_bad) # bad to bad all
      else:
         if none_avalible:
            all_count = attr_val_cnt_all.loc[attr].loc[pd.isnull(attr_val_cnt_all.loc[attr].index)].iloc[0,0] # attr特定类型数量
            good_count = attr_val_cnt_good.loc[attr].loc[pd.isnull(attr_val_cnt_good.loc[attr].index)].iloc[0,0] # attr特定类型good数量
            bad_count = all_count - good_count # attr特定类型bad数量
            attr_count[val] = all_count
            attr_good_count[val] = good_count
            attr_bad_count[val] = bad_count
            all_dist[val] = all_count/float(total_num) # attr样本百分比
            good_dist[val] = good_count/float(total_num_good) # good to good all
            bad_dist[val] = bad_count/float(total_num_bad) # bad to bad all
         else:
            pass
   attr_count = pd.DataFrame(attr_count.items(),columns=[attr,'样本数量']).set_index(attr)
   attr_good_count = pd.DataFrame(attr_good_count.items(),columns=[attr,'好客户数量']).set_index(attr)
   attr_bad_count = pd.DataFrame(attr_bad_count.items(),columns=[attr,'坏客户数量']).set_index(attr)
   all_dist = pd.DataFrame(all_dist.items(),columns=[attr,'样本占比']).set_index(attr)
   good_dist = pd.DataFrame(good_dist.items(),columns=[attr,'好客户占比']).set_index(attr)
   bad_dist = pd.DataFrame(bad_dist.items(),columns=[attr,'坏客户占比']).set_index(attr)
   woe = pd.concat([attr_count,attr_good_count,attr_bad_count,all_dist,good_dist,bad_dist],axis=1)
   try: #good_dist或bad_dist有一个为0则woe break了
      woe['WOE'] = np.log(woe['好客户占比']/woe['坏客户占比'])
      woe['IV'] = np.round((woe['好客户占比']-woe['坏客户占比'])*woe['WOE'],5)
      woe = woe.sort_index(na_position='first')
      total_value = pd.DataFrame([[sum(woe['样本数量']),sum(woe['好客户数量']),sum(woe['坏客户数量']),sum(woe['样本占比']),sum(woe['好客户占比']),sum(woe['坏客户占比']),sum(woe['WOE']),sum(woe['IV'])]],index=['合计'],columns=['样本数量', '好客户数量', '坏客户数量', '样本占比', '好客户占比', '坏客户占比', 'WOE','IV'])
      woe = woe.append(total_value)
   except:
      pass
   return woe
 
def cal_woe(row_index, attr, attr_val_cnt_all, attr_val_cnt_good, none_avalible=True):
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
         attr_count[val] = attr_val_cnt_all.loc[attr,val]['all_count'] # attr特定类型数量
         try:
            attr_good_count[val] = attr_val_cnt_good.loc[attr,val]['good_count'] # attr特定类型good数量
         except:
            attr_good_count[val] = 0
         attr_bad_count[val] = attr_count[val] - attr_good_count[val] # attr特定类型bad数量
         all_dist[val] = attr_count[val]/float(total_num) # attr样本百分比
         good_dist[val] = attr_good_count[val]/float(total_num_good) # good to good all
         bad_dist[val] = attr_bad_count[val]/float(total_num_bad) # bad to bad all
      else:
         if none_avalible:
            all_count = attr_val_cnt_all.loc[attr].loc[pd.isnull(attr_val_cnt_all.loc[attr].index)].iloc[0,0] # attr特定类型数量
            good_count = attr_val_cnt_good.loc[attr].loc[pd.isnull(attr_val_cnt_good.loc[attr].index)].iloc[0,0] # attr特定类型good数量
            bad_count = all_count - good_count # attr特定类型bad数量
            attr_count[val] = all_count
            attr_good_count[val] = good_count
            attr_bad_count[val] = bad_count
            all_dist[val] = all_count/float(total_num) # attr样本百分比
            good_dist[val] = good_count/float(total_num_good) # good to good all
            bad_dist[val] = bad_count/float(total_num_bad) # bad to bad all
         else:
            pass
   temp=attr_count.items()
   attr_row_index=[]
   for j in range(len(temp)):
      attr_row_index.append((temp[j][0],row_index))
   attr_row_index=pd.DataFrame(attr_row_index,columns=[attr,'index']).set_index(attr)
   attr_count = pd.DataFrame(attr_count.items(),columns=[attr,'样本数量']).set_index(attr)
   attr_good_count = pd.DataFrame(attr_good_count.items(),columns=[attr,'好客户数量']).set_index(attr)
   attr_bad_count = pd.DataFrame(attr_bad_count.items(),columns=[attr,'坏客户数量']).set_index(attr)
   all_dist = pd.DataFrame(all_dist.items(),columns=[attr,'样本占比']).set_index(attr)
   good_dist = pd.DataFrame(good_dist.items(),columns=[attr,'好客户占比']).set_index(attr)
   bad_dist = pd.DataFrame(bad_dist.items(),columns=[attr,'坏客户占比']).set_index(attr)
   woe = pd.concat([attr_row_index,attr_count,attr_good_count,attr_bad_count,all_dist,good_dist,bad_dist],axis=1)
   try: #good_dist或bad_dist有一个为0则woe break了
      woe['WOE'] = np.log(woe['好客户占比']/woe['坏客户占比'])
      woe['IV'] = np.round((woe['好客户占比']-woe['坏客户占比'])*woe['WOE'],5)
      woe = woe.sort_index(na_position='first')
      total_value = pd.DataFrame([[sum(woe['index']/len(woe['index'])),sum(woe['样本数量']),sum(woe['好客户数量']),sum(woe['坏客户数量']),sum(woe['样本占比']),sum(woe['好客户占比']),sum(woe['坏客户占比']),sum(woe['WOE']),sum(woe['IV'])]],index=['合计'],columns=['index','样本数量', '好客户数量', '坏客户数量', '样本占比', '好客户占比', '坏客户占比', 'WOE','IV'])
      woe = woe.append(total_value)
   except:
      pass
   return woe
 
def dfs2xlsx(list_dfs, xls_path = None, iv_list=pd.DataFrame()):
    '''将df列表写入excel同一sheet下
    Args:
        list_dfs:dataframe列表
        xls_path:写出的文件路径
    '''
    if xls_path == None:
        xls_path = '~tmp.xlsx'
    writer = ExcelWriter(xls_path)
    i=0
    for df in list_dfs:
        df.to_excel(writer,'Sheet1',startrow = i, encoding="utf8")
        i+= len(df) + 2
    if iv_list.empty!=True:
       iv_list.to_excel(writer, "Sheet2", encoding="utf8")
    writer.save()
 
 
 
def calc_iv(row_index, df_pd, col_rename):
    import pandas as pd
    # attr_list=[col_rename]
    rule_hit_cnt=attr_cnt(df_pd, col_rename)
    rule_hit_good=attr_cnt(df_pd[df_pd['fraudFlag']==0], col_rename)
    # rule_hit_cnt = data.flatMap(lambda x: attr_cnt(x,attr_list)).reduceByKey(add).collect() # 各attr变量不同value数量统计
    # rule_hit_good = data.filter('fraudFlag = 0').flatMap(lambda x: attr_cnt(x,attr_list)).reduceByKey(add).collect() # 各attr变量不同value下target变量为good数量统计
 
    # 转成以各attr,value为index，count为column的DataFrame，方便计算woe、iv
    rule_hit_cnt = pd.DataFrame([(attr,val,count) for ((attr,val),count) in rule_hit_cnt],columns=['attr','value','all_count']).set_index(['attr','value'])
    rule_hit_good = pd.DataFrame([(attr,val,count) for ((attr,val),count) in rule_hit_good],columns=['attr','value','good_count']).set_index(['attr','value'])
    df_woe = cal_woe(row_index, col_rename, rule_hit_cnt, rule_hit_good)
    return df_woe