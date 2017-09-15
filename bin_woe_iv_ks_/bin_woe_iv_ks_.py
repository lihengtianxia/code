import numpy as np
import pandas as pd
import re
import math
import time
import itertools
from itertools import combinations
from numpy import array
from math import sqrt
from multiprocessing import Pool

def get_max_ks(date_df, start, end, rate, bad_name, good_name):
    ks = ''
    if end == start:
        return ks
    bad = date_df.loc[start:end,bad_name]
    good = date_df.loc[start:end,good_name]
    bad_good_cum = list(abs(np.cumsum(bad/sum(bad)) - np.cumsum(good/sum(good))))
    if bad_good_cum:
        ks = start + bad_good_cum.index(max(bad_good_cum))
    return ks

def cut_while_fun(start,end,piece,date_df,rate,bad_name,good_name,counts):
    point_all = []
    if counts >= piece or len(point_all) >= pow(2,piece-1):
        return []
    ks_point = get_max_ks(date_df,start,end,rate,bad_name,good_name)
    if ks_point:
        if ks_point != '':
            t_up = cut_while_fun(start,ks_point,piece,date_df,rate,bad_name,good_name,counts+1)
        else:
            t_up = []
        t_down = cut_while_fun(ks_point+1,end,piece,date_df,rate,bad_name,good_name,counts+1)
    else:
        t_up = []
        t_down = []
    point_all = t_up + [ks_point] + t_down
    return point_all

def ks_auto(date_df,piece,rate,bad_name,good_name):
    t_list = list(set(cut_while_fun(0,len(date_df)-1,piece,date_df,rate,bad_name,good_name,0)))
    ks_point_all = [0] + filter(lambda x: x != '',t_list) + [len(date_df)-1]
    return ks_point_all

def get_combine(t_list, date_df, piece):
    t1 = 0
    t2 = len(date_df)-1
    list0 = t_list[1:len(t_list)-1]
    combine = []
    if len(t_list)-2 < piece:
        c = len(t_list)-2
    else:
        c = piece-1
    list1 = list(itertools.combinations(list0, c))
    if list1:
        combine = map(lambda x: sorted(x + (t1-1,t2)),list1)
    return combine

def cal_iv(date_df,items,bad_name,good_name,rate,total_all):
    iv0 = 0
    total_rate = [sum(date_df.ix[x[0]:x[1],bad_name]+date_df.ix[x[0]:x[1],good_name])*1.0/total_all for x in items]
    if [k for k in total_rate if k < rate]:
        return 0
    bad0 = array(map(lambda x: sum(date_df.ix[x[0]:x[1],bad_name]),items))
    good0 = array(map(lambda x: sum(date_df.ix[x[0]:x[1],good_name]),items))
    bad_rate0 = bad0*1.0/(bad0 + good0)
    if 0 in bad0 or 0 in good0:
        return 0
    good_per0 = good0*1.0/sum(date_df[good_name])
    bad_per0 = bad0*1.0/sum(date_df[bad_name])
    woe0 = map(lambda x: math.log(x,math.e),good_per0/bad_per0)
    if sorted(woe0, reverse=False) == list(woe0) and sorted(bad_rate0, reverse=True) == list(bad_rate0):
        iv0 = sum(woe0*(good_per0-bad_per0))
    elif sorted(woe0, reverse=True) == list(woe0) and sorted(bad_rate0, reverse=False) == list(bad_rate0):
        iv0 = sum(woe0*(good_per0-bad_per0))
    return iv0

def choose_best_combine(date_df,combine,bad_name,good_name,rate,total_all):
    z = [0]*len(combine)
    for i in range(len(combine)):
        item = combine[i]
        z[i] = (zip(map(lambda x: x+1,item[0:len(item)-1]),item[1:]))
    iv_list = map(lambda x: cal_iv(date_df,x,bad_name,good_name,rate,total_all),z)
    iv_max = max(iv_list)
    if iv_max == 0:
        return ''
    index_max = iv_list.index(iv_max)
    combine_max = z[index_max]
    return combine_max

def verify_woe(x):
    if re.match('^\d*\.?\d+$', str(x)):
        return x
    else:
        return 0

def best_df(date_df, items, na_df, factor_name, bad_name, good_name,total_all,good_all,bad_all):
    df0 = pd.DataFrame()
    if items:
        piece0 = map(lambda x: '('+str(date_df.ix[x[0],factor_name])+','+str(date_df.ix[x[1],factor_name])+')',items)
        bad0 = map(lambda x: sum(date_df.ix[x[0]:x[1],bad_name]),items)
        good0 = map(lambda x: sum(date_df.ix[x[0]:x[1],good_name]),items)
        if len(na_df) > 0:
            piece0 = array(list(piece0) + map(lambda x: '('+str(x)+','+str(x)+')',list(na_df[factor_name])))
            bad0 = array(list(bad0) + list(na_df[bad_name]))
            good0 = array(list(good0) + list(na_df[good_name]))
        else:
            piece0 = array(list(piece0))
            bad0 = array(list(bad0))
            good0 = array(list(good0))
        total0 = bad0 + good0
        total_per0 = total0*1.0/total_all
        bad_rate0 = bad0*1.0/total0
        good_rate0 = 1 - bad_rate0
        good_per0 = good0*1.0/good_all
        bad_per0 = bad0*1.0/bad_all
        df0 = pd.DataFrame(zip(piece0,total0,bad0,good0,total_per0,bad_rate0,good_rate0,good_per0,bad_per0),columns=['Bin','Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Good_Rate','Good_Pcnt','Bad_Pcnt'])
        df0 = df0.sort_values(by='Bad_Rate',ascending=False)
        df0.index = range(len(df0))
        bad_per0 = array(list(df0['Bad_Pcnt']))
        good_per0 = array(list(df0['Good_Pcnt']))
        bad_rate0 = array(list(df0['Bad_Rate']))
        good_rate0 = array(list(df0['Good_Rate']))
        bad_cum = np.cumsum(bad_per0)
        good_cum = np.cumsum(good_per0)
        woe0 = map(lambda x: math.log(x, math.e), good_per0/bad_per0)
        if 'inf' in str(woe0):
            woe0 = map(lambda x: verify_woe(x), woe0)
        iv0 = woe0*(good_per0-bad_per0)
        gini = 1-pow(good_rate0,2)-pow(bad_rate0,2)
        df0['Bad_Cum'] = bad_cum
        df0['Good_Cum'] = good_cum
        df0["Woe"] = woe0
        df0["IV"] = iv0
        df0['Gini'] = gini
        df0['KS'] = abs(df0['Good_Cum'] - df0['Bad_Cum'])
    return df0

def all_information(date_df, na_df, piece, rate, factor_name, bad_name, good_name,total_all,good_all,bad_all):
    p_sort = range(piece+1)
    p_sort.sort(reverse=True)
    t_list = ks_auto(date_df,piece,rate,bad_name,good_name)
    if not t_list:
        df1 = pd.DataFrame()
        print 'Warning: this data cannot get bins or the bins does not satisfy monotonicity'
        return df1
    df1 = pd.DataFrame()
    for c in p_sort[:piece-1]:
        combine = get_combine(t_list,date_df,c)
        best_combine = choose_best_combine(date_df,combine,bad_name,good_name,rate,total_all)
        df1 = best_df(date_df,best_combine,na_df,factor_name,bad_name,good_name,total_all,good_all,bad_all)
        if len(df1) != 0:
            gini = sum(df1['Gini']*df1['Total_Num']/sum(df1['Total_Num']))
            print 'piece_count:',str(len(df1))
            print 'IV_All_Max:',str(sum(df1['IV']))
            print 'Best_KS:',str(max(df1['KS']))
            print 'Gini_index:',str(gini)
            print df1
            return df1
    if len(df1) == 0:
        print 'Warning: this data cannot get bins or the bins does not satisfy monotonicity'
        return df1

def verify_factor(x):
    if x in ['NA', 'NAN', '', ' ', 'MISSING','NONE','NULL']:
        return 'NAN'
    if re.match('^\-?\d*\.?\d+$',x):
        x = float(x)
    return x

def path_df(path,sep,factor_name):
    data = pd.read_csv(path,sep=sep)
    data[factor_name] = data[factor_name].astype(str).map(lambda x: x.upper())
    data[factor_name] = data[factor_name].apply(lambda x: re.sub(' ','MISSING',x))
    return data

def verify_df_multiple(date_df,factor_name,total_name,bad_name,good_name):
    """
    :param date_df: factor_name,...
    :return: factor_name,good_name,bad_name
    """
    date_df = date_df.fillna(0)
    cols = date_df.columns
    if total_name in cols:
        date_df = date_df[date_df[total_name] != 0]
        if bad_name in cols and good_name in cols:
            date_df_check = date_df[date_df[good_name] + date_df[bad_name] - date_df[total_name] != 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print 'Error: total amounts is not equal to the sum of bad & good amounts'
                print date_df_check
                return date_df
        elif bad_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[bad_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print 'Error: total amounts is smaller than bad amounts'
                print date_df_check
                return date_df
            date_df[good_name] = date_df[total_name] - date_df[bad_name]
        elif good_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[good_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print 'Error: total amounts is smaller than good amounts'
                print date_df_check
                return date_df
            date_df[bad_name] = date_df[total_name] - date_df[good_name]
        else:
            print 'Error: lack of bad or good data'
            date_df = pd.DataFrame()
            return date_df
        del date_df[total_name]
    elif bad_name not in cols:
        print 'Error: lack of bad data'
        date_df = pd.DataFrame()
        return date_df
    elif good_name not in cols:
        print 'Error: lack of good data'
        date_df = pd.DataFrame()
        return date_df
    date_df[good_name] = date_df[good_name].astype(int)
    date_df[bad_name] = date_df[bad_name].astype(int)
    date_df = date_df[date_df[bad_name] + date_df[good_name] != 0]
    date_df[factor_name] = date_df[factor_name].map(verify_factor)
    date_df = date_df.sort_values(by=[factor_name],ascending=True)
    date_df[factor_name] = date_df[factor_name].astype(str)
    if len(date_df[factor_name]) != len(set(date_df[factor_name])):
        df_bad = date_df.groupby(factor_name)[bad_name].agg([(bad_name,'sum')]).reset_index()
        df_good = date_df.groupby(factor_name)[good_name].agg([(good_name,'sum')]).reset_index()
        good_dict = dict(zip(df_good[factor_name],df_good[good_name]))
        df_bad[good_name] = df_bad[factor_name].map(good_dict)
        df_bad.index = range(len(df_bad))
        date_df = df_bad
    return date_df

def verify_df_two(date_df,flag_name,factor_name,good_name,bad_name):
    """
    :param date_df: factor_name,flag_name
    :return: factor_name,good_name,bad_name
    """
    date_df = date_df.drop(date_df[date_df[flag_name].isnull()].index)
    if len(date_df) == 0:
        print 'Error: the data is wrong'
        return date_df
    check = date_df[date_df[flag_name] > 1]
    if len(check) != 0:
        print 'Error: there exits the number bigger than one in the data'
        date_df = pd.DataFrame()
        return date_df
    if flag_name != '':
        try:
            date_df[flag_name] = date_df[flag_name].astype(int)
        except:
            print 'Error: the data is wrong'
            date_df = pd.DataFrame()
            return date_df
    date_df = date_df[flag_name].groupby([date_df[factor_name],date_df[flag_name]]).count().unstack().reset_index().fillna(0)
    date_df.columns = [factor_name,good_name,bad_name]
    date_df[factor_name] = date_df[factor_name].map(verify_factor)
    date_df = date_df.sort_values(by=[factor_name],ascending=True)
    date_df.index = range(len(date_df))
    date_df[factor_name] = date_df[factor_name].astype(str)
    return date_df

def universal_df(data,flag_name,factor_name,total_name,bad_name,good_name):
    if flag_name != '':
        data = data[[factor_name,flag_name]]
        data = verify_df_two(data,flag_name,factor_name,good_name,bad_name)
    else:
        data = verify_df_multiple(data,factor_name,total_name,bad_name,good_name)
    return data

def Best_KS_Bin(path='',data=pd..DataFrame(),sep=',',flag_name='',factor_name='name',total_name='total',bad_name='bad',good_name='good',piece=5,rate=0.05,not_in_list=[],value_type=True):
    """
    :param value_type: True is numerical; False is nominal
    """
    time0 = time.time()
    none_list = ['NA', 'NAN', '', ' ', 'MISSING','NONE','NULL']
    if path != '':
        data = path_df(path,sep,factor_name)
    elif len(data) == 0:
        print 'Error: there is no data'
        return data
    data[factor_name] = data[factor_name].map(lambda x: unicode(x).upper())
    data = universal_df(data,flag_name,factor_name,total_name,bad_name,good_name)
    if len(data) == 0:
        return data
    good_all = sum(data[good_name])
    bad_all = sum(data[bad_name])
    total_all = good_all + bad_all
    if not_in_list:
        not_name = [unicode(k).upper()for k in not_in_list]
        for n0 in none_list:
            if n0 in not_name:
                not_name += ['NAN']
                break
        na_df = data[data[factor_name].isin(not_name)]
        if (0 in na_df[good_name]) or (0 in na_df[bad_name]):
            not_value = list(set(list(na_df[na_df[good_name] == 0][factor_name]) + list(na_df[na_df[bad_name] == 0][factor_name])))
            na_df = na_df..drop(na_df[na_df[factor_name].isin(not_value)].index)
            na_df.index = range(len(na_df))
        not_list = list(set(na_df[factor_name]))
        date_df = data.drop(data[data[factor_name].isin(not_list)].index)
    else:
        na_df = pd.DataFrame()
        date_df = data
    if len(date_df) == 0:
        print 'Error: the data is wrong.'
        data = pd.DataFrame()
        return data
    if value_type:
        date_df[factor_name] = date_df[factor_name].map(verify_factor)
        type_len = set([type(k) for k in list(date_df[factor_name])])
        if len(type_len) > 1:
            other_df = date_df[date_df[factor_name].map(lambda x: type(x) == str)]
            date_df = date_df[date_df[factor_name].map(lambda x: type(x) == float)]
            date_df = date_df.sort_values(by=factor_name)
            other_df = other_df.sort_values(by=factor_name)
            date_df = other_df.append(date_df)
        else:
            date_df = date_df.sort_values(by=factor_name)
    else:
        date_df['bad_rate'] = date_df[bad_name]*1.0/(date_df[good_name]+date_df[bad_name])
        date_df = date_df.sort_values(by=['bad_rate',factor_name],ascending=False)
    date_df[factor_name] = date_df[factor_name].astype(str)
    date_df.index = range(len(date_df))

    bin_df = all_information(date_df,na_df,piece,rate,factor_name,bad_name,good_name,total_all,good_all,bad_all)
    time1 = time.time()
    print 'spend time(s):', round(time1-time0,0)
    return bin_df
