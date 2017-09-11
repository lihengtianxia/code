#IV_WOE计算代码更新，只要能将spark数据集转换成pandas，无论多少字段都可以快速分bin。分类型字段只能按照现有水平个数计算IV和WOE，不能进行自动分bin处理。

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
 
 
cols=[
    #添加需要被分bin的字段
    ]
import numpy as np
for col in cols:
    col_rename=col+"_Q"
    print col_rename
    if df_pd[col].dtype==type(1.0) or df_pd[col].dtype==type(1):
        for num_of_bins in range(2,10):
            temp=df_pd[col]
            if len(temp[temp>0])>0:
                cutspoint=[-0.1]
                temp=sorted(temp[temp>0])
                bin_num=len(temp)/num_of_bins
                for i in range(0,num_of_bins):
                    cutspoint.append(round(temp[i*bin_num]))
                cutspoint.append(temp[len(temp)-1]+1)
                cutspoint = list(set(cutspoint))
                cutspoint.sort()
                cut_list.append(cutspoint)
                try:
                    df_pd[col_rename]=pd.Series(pd.cut(df_pd[col], cutspoint,labels=[i for i in cutspoint[0:len(cutspoint)-1]]), index=df_pd.index)
                    df_pd[col_rename]= df_pd[col_rename].replace(np.nan, 0)
                    # cutspoint.append((col,cutspoint))
                    # data=sqlContext.createDataFrame(df_pd)
                    # iv, ivtable=calc_iv(data.filter(col+"<="+str(out_value)),col_rename)
                    # ivs.append((col,len(df_pd[col_rename].unique()),iv))
                    ivtable=calc_iv(row_index, df_pd,col_rename)
                    x=ivtable.reset_index()
                    old_str=str(x[x[col_rename].ne("合计")]["WOE"])
                    # Sorted by WOE
                    new_str=str(x[x[col_rename].ne("合计")].sort_values(by="WOE", ascending=0)["WOE"])
                    ivtables.append(ivtable)
                    row_index+=1
                    if old_str==new_str:
                        final_cutspoint=cutspoint
                        final_ivtable=ivtable
                        # opt_flag=True
                except Exception as e:
                    print str(e)+col_rename
                continue
                print col_rename
                if final_cutspoint:
                    df_pd[col_rename]=pd.Series(pd.cut(df_pd[col], final_cutspoint,labels=[i for i in final_cutspoint[0:len(final_cutspoint)-1]]), index=df_pd.index)
                else:
                    df_pd[col_rename]=pd.Series(pd.cut(df_pd[col], cutspoint,labels=[i for i in cutspoint[0:len(cutspoint)-1]]), index=df_pd.index)
    else:
        ivtable=calc_iv(row_index,df_pd,col)
        ivtables.append(ivtable)
        row_index+=1
 
 
ivs=[]
for df in ivtables:
   ivs.append((df.index.name, df["IV"][df["IV"]<np.inf][df["IV"]>-np.inf].sum()/2))
 
iv_df=pd.DataFrame(ivs,columns=["col", "iv"]).groupby("col").max().sort_values("iv", ascending=False)
 
import time
partner=''
dfs2xlsx(ivtables, '/home/chao.qin/'+partner+'_iv'+time.strftime('%Y%m%d', time.localtime(time.time()))+'.xlsx', iv_df)