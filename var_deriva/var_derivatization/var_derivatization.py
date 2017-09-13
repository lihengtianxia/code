#coding:utf-8
from pyspark import SparkContext
from pyspark import HiveContext
import datetime
import numpy as np
from numpy import array
import pandas as pd
import os
import pickle
import re
import math
from pyspark.sql import Row
from pyspark.sql import SparkSession
from dateutil.parser import parse
import time
import os
import pickle
import sys
import json
import MySQLdb
from collections import Counter
reload(sys)
sys.setdefaultencoding('utf8')
hc=SparkSession.builder.appName('appName').enableHiveSupport().getOrCreate()
sc=hc.sparkContext

conn=MySQLdb.connect(host='192.168.100.41',user='shuo.gan',passwd='123456',db='forseti',charset='utf8')
cur=conn.cursor()
aa=cur.execute('select * from admin_partner')
info=cur.fetchmany(aa)
cur.close()
conn.close()
ind_map_1=dict(map(lambda x:[x[4],x[19]],filter(lambda y:y[19] not in ['0','44000','Test',None,'46000','yuankui-type','49000'],info)))
ind_map_2=dict(map(lambda x:[x[4],x[21]],filter(lambda y:y[21] not in [None],info)))
ind_dict={}
for key in ['o2o','Imbank','IFinanceWeb','Fund','Bank','Insurance','Crowdfunding','ElectronicBusiness']:
    ind_dict[key]=list(set(map(lambda x:x[4],filter(lambda y:y[19]==key,info))))
for key in ['offloan','Micloan','P2pweb','Unconsumerfinance','Inconsumerfinance','consumerfinance','Manucarfinance','onlinebank']:
    ind_dict[key]=list(set(map(lambda x:x[4],filter(lambda y:y[21]==key,info))))
ind_dict['finance']=list(set(map(lambda x:x[4],filter(lambda y:y[19] in ['Imbank','Bank','Insurance'],info))))
ind_dict['con']=list(set(map(lambda x:x[4],filter(lambda y:y[21] in ['Inconsumerfinance','Unconsumerfinance','consumerfinance'],info))))
ind_dict['Microloan']=list(set(map(lambda x:x[4],filter(lambda y:y[21] in ['offloan','Micloan'],info))))
ind_dict['all']=list(set(map(lambda x:x[4],info)))
f=open('match/carnumber.pkl')
Bin_dict=pickle.load(f)
f.close()


time_cut_list=['365day','180day','90day','60day','30day','15day','30_90day','90_180day','180_365day']
dev_list=['ipaddressprovince','ipaddresscity','trueipaddressprovince','trueipaddresscity','canvas','accountemail','mobileaddresscity','mobileaddressprovince','qqnumber','cardnumber','deviceid','browser','apptype','payeeidnumber','isemulator','isroot']
area_dic={11:u'BJ',12:u'TJ',13:u'HE',14:u'SX',15:u'IM',21:u'LN',22:u'JL',23:u'HL',31:u'SH',32:u'JS',33:u'ZJ',34:u'AH',35:u'FJ',36:u'JX',37:u'SD',
          41:u'HA',42:u'HB',43:u'HN',44:u'GD',45:u'GX',46:u'HI',50:u'CQ',51:u'SC',52:u'GZ',53:u'YN',54:u'XZ',
          61:u'SN',62:u'GS',63:u'QH',64:u'NX',65:u'XJ',71:u'TW',81:u'HK',82:u'MO'}

#相应条件下不同平台、不同记录的相关统计：
#个数、比例
def count_partner_record_map(stats,val,event,ind,time_cut):
    value_stats={}
    val_list=map(lambda x:x['partnercode'],val)
    stats_list=map(lambda x:x['partnercode'],stats)
    value_stats['freq_record2_'+ event + '_'+ ind + '_'+time_cut]=round(len(set(stats_list))*1.0/len(set(val_list)),4)
    value_stats['freq_record2_' + event + '_' + ind + '_' + time_cut] = len(stats_list)
    if val:
        value_stats['ratio_cnt_partner2_'+event + '_' + ind + '_' + time_cut]=round(len(set(stats_list))*1.0/len(set(val_list)),4)
        value_stats['ratio_freq_partner2_' + event + '_' + ind + '_' + time_cut] = round(len(stats_list) * 1.0 / len(val_list), 4)

    else:
        value_stats['ratio_cnt_partner2_' + event + '_' + ind + '_' + time_cut]=-1111
        value_stats['ratio_freq_partner2_' + event + '_' + ind + '_' + time_cut]=-1111

    return value_stats

#(2) 日最大值、平均值
def max_daily_parnter_record_map(value,event,ind,time_cut):
    value_stats={}
    list1=map(lambda x:(x['partnercode'],int(float(x['eventoccurtime']))/86400),value)
    dict1=Counter(x[1] for x in list1)
    value_stats['max_freq_record_daily2_'+event+'_'+ind+'_'+time_cut]=int(max(dict1.values() or [0]))
    value_stats['mean_freq_record_daily2_'+ event+'_'+ind+'_'+time_cut]=int(np.mean(dict1.values()) or [0])
    dict2=Counter([x[1] for x in set(list1)])
    value_stats['max_cnt_partner_daily2_' +event+'_'+ind+'_'+time_cut]=int(max(dict2.values() or [0]))
    value_stats['mean_cnt_partner_daily2_' + event + '_' + ind + '_' + time_cut] = int(np.mean(dict2.values() or [0]))
    return value_stats
##时间戳时间相关计算
#相应条件下特殊时间段相关计算
def special_time_map(stats,event,ind,time_cut):
    value_stats={}
    time_list=map(lambda x:float(x['eventoccurtime']),stats)
    weekend=filter(lambda x:time.localtime(x)[6] in [5,6],time_list)
    weekday=[x for x in time_list if x not in weekend]
    night=filter(lambda x:time.localtime(x)[3] in range(0,9),time_list)
    day=[x for x in time_list if x not in night]
    value_stats['freq_night2_' +event + '_' + ind + '_'+ time_cut]=len(night)
    value_stats['freq_day2_' + event + '_' + ind + '_' + time_cut] =len(day)
    value_stats['freq_weekday2_' + event + '_' + ind + '_' + time_cut] = len(weekday)
    value_stats['freq_weekend2_' + event + '_' + ind + '_' + time_cut] = len(weekend)
    if time_list:
        value_stats['ratio_freq_night2_' + event + '_' + ind + '_' + time_cut]=round(len(night)*1.0/len(time_list),4)
        value_stats['ratio_freq_day2_' + event + '_' + ind + '_' + time_cut] = round(len(day) * 1.0 / len(time_list), 4)
        value_stats['ratio_freq_weekday2_' + event + '_' + ind + '_' + time_cut] = round(len(weekday) * 1.0 / len(time_list), 4)
        value_stats['ratio_freq_weekend2_' + event + '_' + ind + '_' + time_cut] = round(len(weekend) * 1.0 / len(time_list), 4)

    else:
        value_stats['ratio_freq_night2_' + event + '_' + ind + '_' + time_cut] = -1111
        value_stats['ratio_freq_day2_' + event + '_' + ind + '_' + time_cut] = -1111
        value_stats['ratio_freq_weekday2_' + event + '_' + ind + '_' + time_cut] =-1111
        value_stats['ratio_freq_weekend2_' + event + '_' + ind + '_' + time_cut] =-1111
    return value_stats
#相应条件下时间长度相关计算
def time_length_map(stats,nowTimeLinux,event,ind,time_cut):
    value_stats={}
    time_length={'15day':1296000,'365day':31536000,'180day':15552000,'90day':7776000,'60day':5184000,'30day':2592000,'30_90day':5184000,'90_180day':7776000,'180_365day':15984000}
    value_stats['length_first2_' + event +'_'+ ind + '_' + time_cut]=-1111
    value_stats['length_last2_' + event + '_' + ind + '_' + time_cut] = -1111
    value_stats['length_first_last2_' + event +'_'+ ind + '_' + time_cut]=-1111
    value_stats['length_time2_' + event + '_' + ind + '_' + time_cut] = -1111
    value_stats['length_event2_' + event + '_' + ind + '_' + time_cut] = -1111
    if stats:
        time_list=map(lambda x:float(x['eventoccurtime']),stats)
        time0=min(time_list)
        time1=max(time_list)
        time_inter0=round((nowTimeLinux-time0)*1.0/86400,2)
        time_inter1 = round((nowTimeLinux - time1) * 1.0 / 86400, 2)
        value_stats['length_first2_' + event +'_'+ ind + '_' + time_cut]=time_inter0
        value_stats['length_last2_' + event + '_' + ind + '_' + time_cut] = time_inter1
        value_stats['mean_length_time2_' + event + '_' + ind + '_' + time_cut] =round(time_length[time_cut]*1.0/(len(time_list)+1)/86400,4)
        if len(time_list)>1:
            value_stats['length_first_last2_' + event +'_'+ ind + '_' + time_cut]=round((time1-time0)*1.0/86400,4)
            value_stats['mean_length_time2_' + event + '_' + ind + '_' + time_cut] = round((time1 - time0) * 1.0 / (len(time_list)+1)/86400, 4)

    return value_stats

#############################################event+all+time_cut
#相应条件下不同1,2级行业的个数、比例
def ind_map(val1,val,event,time_cut):
    value_stats={}
    cnt_ind1=map(lambda x:ind_map_1[x['partnercode']],filter(lambda x:x['partnercode'] in ind_map_1,val1))
    cnt_ind2 = map(lambda x: ind_map_2[x['partnercode']], filter(lambda x: x['partnercode'] in ind_map_2, val1))
    value_stats['cnt_indone2_'+event+'_all_'+time_cut]=len(set(cnt_ind1))
    value_stats['cnt_indtwo2_' + event + '_all_' + time_cut] = len(set(cnt_ind2))
    return value_stats


###############################################all+all+time_cut
#金额类相关变量
def check_number(x):
    try:
        f=float(x)
        return f
    except:
        return None

def payamount_map(value,time_cut):
    value_stats={}
    value_stats['max_payamount2_all_all_'+time_cut]=-1111
    value_stats['min_payamount2_all_all_' + time_cut] = -1111
    value_stats['mean_payamount2_all_all_' + time_cut] = -1111
    value_stats['ratio_freq_payamount2_all_all_' + time_cut] = -1111
    payamount_list=map(lambda x:float(x['payamount']),filter(lambda x:x.get('payamount') and check_number(x['payamount']),value))
    value_stats['ratio_freq_payamount2_all_all_' + time_cut]=len(payamount_list)
    if payamount_list:
        value_stats['max_payamount2_all_all_'+time_cut]=max(payamount_list)
        value_stats['min_payamount2_all_all_' + time_cut] = min(payamount_list)
        value_stats['mean_payamount2_all_all_' + time_cut] = np.mean(payamount_list)
        value_stats['ratio_freq_payamount2_all_all_' + time_cut] = round(len(payamount_list)*1.0/len(value),4)
    return value_stats

#policy decision 的个数及ratio统计
def policy_stats_fun(value,time_cut):
    value_stats={}
    decision_name=['Accept','Review','Reject']
    d_list=[]
    val=map(lambda x:eval(x['policydecision']),filter(lambda x:x.get('policydecision'),value))
    for val0 in val:
        d_list+=val0
    if d_list:
        for i in range(3):
            dname=decision_name[i]
            count0=d_list.count(str(i))
            value_stats['freq_policy_'+str(i)+'2_all_all_'+time_cut]=count0
            value_stats['ratio_freq_policy_'+str(i)+'2_all_all_'+time_cut]=round(count0*1.0/len(d_list),4)
    else:
        dics0=dict(zip(map(lambda x:'freq_policy_'+str(x)+'2_all_all_'+time_cut,decision_name),[0]*len(decision_name)))
        dics1 = dict(zip(map(lambda x: 'ratio_freq_policy_' + str(x) + '2_all_all_' + time_cut, decision_name),[-1111] * len(decision_name)))
        value_stats.update(dics0)
        value_stats.update(dics1)
    return value_stats


##riskstatus的个数及ratio统计
def riskstatus_fun(value,time_cut):
    value_stats={}
    decision_list=['Reject','Review','Accept']
    d_list=[]
    val=map(lambda x:x['riskstatus'],filter(lambda x:x.get('riskstatus'),value))
    for val0 in val:
        d_list+=val0
    if d_list:
        for d in decision_list:
            count0=d_list.count(d)
            value_stats['freq_risk_'+str(d)+'2_all_all_'+time_cut]=count0
            value_stats['ratio_freq_risk_' + str(d) + '2_all_all_' + time_cut] = round(count0*1.0/len(d_list),4)
    else:
        dics0 = dict(zip(map(lambda x: 'freq_risk_' + str(x) + '2_all_all_' + time_cut, decision_list),[0] * len(decision_list)))
        dics1 = dict(zip(map(lambda x: 'ratio_freq_risk_' + str(x) + '2_all_all_' + time_cut, decision_list),[-1111] * len(decision_list)))
        value_stats.update(dics0)
        value_stats.update(dics1)

    return value_stats



##设备类相关信息
#device相关信息

def device_stats(value,time_cut):
    value_stats={}
    for col in dev_list:
        d_list=map(lambda x:x[col],filter(lambda x:x.get(col),value))
        value_stats['cnt_'+col+'2_all_all_'+time_cut]=len(set(d_list))
        value_stats['freq_' + col + '2_all_all_' + time_cut] = len(d_list)
    return value_stats

def score_map(value,time_cut):
    value_stats={}
    for score in ['riskscore']:
        d_list=map(lambda x:x[score],filter(lambda x:x.get(score),value))
        value_stats['max_'+score+'2_all_all_'+time_cut]=-1111
        value_stats['min_' + score + '2_all_all_' + time_cut] = -1111
    if d_list:
        value_stats['max_'+score+'2_all_all_'+time_cut]=max(map(lambda x:float(x),d_list))
        value_stats['min_' + score + '2_all_all_' + time_cut] = max(map(lambda x: float(x), d_list))
    return value_stats

#系统相关信息
def check_system(x):
    if x=='WINDOWS PHONE':
        return 'WP'
    elif x in ['IPOD','IPHONE','IPAD']:
        return 'IOS'

    elif re.search('IOS',x):
        return 'IOS'
    elif re.search('WINDOWS',x):
        return 'WINDOWS'
    elif re.search('MAC',x):
        return 'MACOS'
    elif re.search('LINUX',x):
        return 'linux'
    elif re.search('ANDROID',x):
        return 'Android'
    else:
        return 'Other'

def devitype_map(value,time_cut):
    value_stats={}
    sys_list=['Windows','MacOS','IOS','Android','WP','Linux']
    list1=map(lambda x:x['devicetype'].upper(),filter(lambda x:x.get('devicetype'),value))
    list2=[check_system(x) for x in list1]
    value_stats['cnt_sys_all_all_'+time_cut]=len(set(list2))
    for system in sys_list:
        value_stats['freq_'+'2_all_all_'+time_cut]=list2.count(system)
    return  value_stats

#IP地址相关信息
def check_ip(x):
    if x:
        ip_list=[y for y in x.split(',') if y[:6]!='_TDERR']
    else:
        ip_list=[]
    return ip_list

def ip_map(value,time_cut):
    value_stats={}
    ip_list=map(lambda x:check_ip(x.get('ipaddress')),value)
    tp_list=map(lambda x:check_ip(x.get('trueipaddress')),value)
    if ip_list:
        ip_list=reduce(lambda x,y:x+y,ip_list)
    if tp_list:
        tp_list=reduce(lambda x,y:x+y,tp_list)
    value_stats['cnt_ipaddress2_all_all_' +time_cut]=len(set(ip_list))
    value_stats['cnt_trueipaddress2_all_all_' + time_cut] = len(set(ip_list))
    value_stats['freq_ipaddress2_all_all_' + time_cut] = len(map(lambda x:x['ipaddress'],filter(lambda y:y.get('ipaddress'),value)))
    value_stats['freq_trueipaddress2_all_all_' + time_cut] = len(map(lambda x: x['trueipaddress'], filter(lambda y: y.get('trueipaddress'), value)))
    value_stats['freq_diff_trueaddress2_trueipaddress2_all_all_' + time_cut] = len(filter(lambda x:check_ip(x.get('trueipaddress'))not in check_ip(x.get('ipaddress')),value))
    return value_stats

#3 银行卡相关信息
def cardnumber(value,time_cut):
    value_stats={}
    Card_list=map(lambda x:x['cardnumber'],filter(lambda x:x.get('cardnumber'),value))
    BankCard_list=[x for x in Card_list if len(x) in [14,15,16,17,18,19]]
    class_list=[]
    if BankCard_list:
        for card in set(BankCard_list):
            for i in set(range(2,11)):
                Bin=card[:i]
                if Bin in set(Bin_dict.keys()) and len(card)==Bin_dict[Bin]['length_card']:
                    class_list.append(Bin_dict[Bin]['card_class'])
    value_stats['cnt_SemiCreditCard2_all_all_'+time_cut]=class_list.count('SemiCreditCard')
    value_stats['cnt_CreditCard2_all_all_' + time_cut] = class_list.count('SemiCreditCard')
    value_stats['cnt_DebitCard2_all_all_' + time_cut] = class_list.count('SemiCreditCard')
    return value_stats
##################################################################################
###id独有
#解析身份证信息
def get_age(uid,nowTimeLinux):
    length=len(uid)
    if length==18 and uid[6:10].isdigit():
        age=time.localtime(nowTimeLinux[0]-int(uid[6:10]))
    elif length==15 and uid[6:8].isdigit():
        age=time.localtime((nowTimeLinux[0])-1900-int(uid[6:8]))
    else:
        age=-1111
    return age

def get_gender(uid):
    length=len(uid)
    if length==18 and uid[16].isdigit():
        gender=int(uid[16])%2
    elif length==15 and uid[14].isdigit():
        gender=int(uid[14])%2
    else:
        gender=-1111
    return gender
def get_province(uid):
    if uid[:2].isdigit() and int(uid[:2])in area_dic:
        province=area_dic[int(uid[:2])]
    else:
        province=-1111
    return province

def info_from_id(x,nowTimeLinux):
    value_stats={}
    uid=unicode(x.uid)
    value_stats['age2']=get_age(uid,nowTimeLinux)
    value_stats['gender']=get_gender(uid)
    value_stats['province2']=get_province(uid)
    return value_stats

def stats_for_all_fun(value,nowTimeLinux):
    value_stats={}
    for time_cut in set(time_cut_list):
        val=value['all_all_'+time_cut]
        value_stats.update(payamount_map(val,time_cut))
        value_stats.update(policy_stats_fun(val, time_cut))
        value_stats.update(riskstatus_fun(val, time_cut))
        value_stats.update(device_stats(val, time_cut))
        value_stats.update(score_map(val, time_cut))
        value_stats.update(devitype_map(val, time_cut))
        value_stats.update(ip_map(val, time_cut))
        value_stats.update(cardnumber(val, time_cut))
        for event in set(['all','Loan','Register','Login','Trade','Apply']):
            val1=value[event+'_all_'+time_cut]
            value_stats.update(ind_map(val1,val,event,time_cut))
            for ind in set(ind_dict.keys()):
                stats=value[event+'_'+ind+'_'+time_cut]
                value_stats.update(count_partner_record_map(stats,val,event,ind,time_cut))
                value_stats.update(max_daily_parnter_record_map(stats, val, event, ind, time_cut))
                value_stats.update(special_time_map(stats, val, event, ind, time_cut))
                value_stats.update(time_length_map(stats, val, event, ind, time_cut))

    return value_stats
#############################################################################################################################
#汇总
def stats_fun_for_all_value(x):
    nowTimeLinux=float(x.loan_date_unix)
    id_stats={}
    mobile_stats={}
    for time_cut in set(time_cut_list):
        id_stats['all_all_'+time_cut]=x['id_value_time_cut'][time_cut]
        mobile_stats['all_all_'+time_cut]=x['mobile_value_time_cut'][time_cut]
        for event in set(['Loan','Register','Login','Trade','Apply']):
            id_stats[event+'_all'+time_cut]=[x1 for x1 in id_stats['all_all_'+time_cut] if x1['eventtype']==event]
            mobile_stats[event+'_all'+time_cut]=[x2 for x2 in mobile_stats['all_all_'+time_cut] if x2['eventtype']==event]
            for ind in set(ind_dict.keys()):
                id_stats[event+'_'+ind+'_'+time_cut]=[y1 for y1 in id_stats[event+'_all_'+time_cut]if y1['partnercode'] in ind_dict[ind]]
                mobile_stats[event +'_' +ind+'_'+time_cut] = [y2 for y2 in mobile_stats[event + '_all_' + time_cut] if y2['partnercode'] in ind_dict[ind]]
        for ind in set(ind_dict.keys()):
            id_stats['all_'+ind+'_'+time_cut]=[y1 for y1 in id_stats['all_all_'+time_cut] if y1['partnercode'] in ind_dict[ind]]
            mobile_stats['all_' + ind + '_' + time_cut] = [y2 for y2 in mobile_stats['all_all_' + time_cut] if y2['partnercode'] in ind_dict[ind]]
    id_value_stats = info_from_id(x,nowTimeLinux)
    mobile_value_stats={}
    if x.uid:
        id_value_stats.update(stats_for_all_fun(id_stats,nowTimeLinux))
    if x.umobile:
        mobile_value_stats.update(stats_for_all_fun(mobile_stats,nowTimeLinux))
    value=x.asDict()
    value['id_value_stats']=dict(zip(id_value_stats.keys(),map(lambda x:unicode(x),id_value_stats.values())))
    value['mobile_value_stats'] = dict(zip(mobile_value_stats.keys(), map(lambda x: unicode(x), mobile_value_stats.values())))
    del value['id']
    del value['mobile']
    del value['loan_date_unix']
    del value['id_value_time_cut']
    del value['mobile_value_time_cut']
    return Row(**value)


def map_time_all(x):
    d_list=[]
    m_list=[]
    id_value_time_cut={'15day':[],'365day':[],'180day':[],'90day':[],'60day':[],'30day':[],'30_90day':[],'90_180day':[],'180_365day':[]}
    mobile_value_time_cut={'15day':[],'365day':[],'180day':[],'90day':[],'60day':[],'30day':[],'30_90day':[],'90_180day':[],'180_365day':[]}
    if x.id_value:
        for l in x.id_value:
            check_num=0
            if l['eventtype']!='Loan':
                check_num=1
            else:
                key='Loan'+str(int(float(l['eventoccurtime']))/86400)+l['partnercode']
                if key not in d_list:
                    check_num=1
                    d_list.append(key)
            if check_num==1 and float(l['eventoccurtime'])<=float(x.loan_date_unix):
                if float(x.loan_date_unix)-float(l['eventocurtime'])<31536000:
                    id_value_time_cut['365day'].append()
                if float(x.loan_date_unix) - float(l['eventocurtime']) < 15552000:
                    id_value_time_cut['180day'].append()
                if float(x.loan_date_unix)-float(l['eventocurtime'])<7776000:
                    id_value_time_cut['90day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])<5184000:
                    id_value_time_cut['60day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])<2592000:
                    id_value_time_cut['30day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])<1296000:
                    id_value_time_cut['15day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])>=2592000 and (float(x.loan_date_unix)-float(l['eventoccurtime'])<7776000):
                    id_value_time_cut['30_90day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])>=7776000 and (float(x.loan_date_unix)-float(l['eventoccurtime'])<15552000):
                    id_value_time_cut['90_180day'].append()

                if float(x.loan_date_unix)-float(l['eventocurtime'])>=15552000 and (float(x.loan_date_unix)-float(l['eventoccurtime'])<31536000):
                    id_value_time_cut['180_365day'].append()

    if x.mobile_value:
        for l in x.mobile_value:
            check_num=0
            if l['eventype']!='Loan':
                check_num=1
            else:
                key='Loan'+str(int(float(l['eventoccurtime']))/86400)+l['partnercode']
                if key not in m_list:
                    check_num=1
                    m_list.append(key)
            if check_num==1 and float(l['eventoccurtime'])<=float(x.loan_date_unix):
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<31536000:
                    mobile_value_time_cut['365day'].append(l)
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<15552000:
                    mobile_value_time_cut['180day'].append(l)
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<7776000:
                    mobile_value_time_cut['90day'].append(l)
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<5184000:
                    mobile_value_time_cut['60day'].append(l)
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<2592000:
                    mobile_value_time_cut['30day'].append(l)
                if float(x.loan_date_unix)-float(l['eventoccurtime'])<1296000:
                    mobile_value_time_cut['15day'].append(l)
                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 2592000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 7776000):
                    id_value_time_cut['30_90day'].append(l)

                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 7776000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 15552000):
                    id_value_time_cut['90_180day'].append(l)

                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 15552000 and ( float(x.loan_date_unix) - float(l['eventoccurtime']) < 31536000):
                    id_value_time_cut['180_365day'].append(l)

    value=x.asDict()
    value['id_value_time_cut']=id_value_time_cut
    value['mobile_value_time_cut']=mobile_value_time_cut
    del value['id_value']
    del value['mobile_value']
    return Row(**value)



###单跑身份证
def stats_fun_for_id_value(x):
    nowTimeLinux=float(x.loan_date_unix)
    id_stats={}
    for time_cut in set(time_cut_list):
        id_stats['all_all_'+time_cut]=x['id_value_time_cut'][time_cut]
        for event in set(['Loan','Register','Login','Trade','Apply']):
            id_stats[event+'_all_'+time_cut]=[x1 for x1 in id_stats['all_all_'+time_cut]if x1['eventtype']==event]
            for ind in set(ind_dict.keys()):
                id_stats[event+'_'+ind+'_'+time_cut]=[y1 for y1 in id_stats[event+'_all_'+time_cut]if y1['partnercode']in ind_dict[ind]]
        for ind in set(ind_dict.keys()):
            id_stats['all_'+ind+'_'+time_cut]=[y1 for y1 in id_stats['all_all_'+time_cut]if y1['partnercode']in ind_dict[ind]]
        id_value_stats=info_from_id(x,nowTimeLinux)
        if x.id:
            id_value_stats.update(stats_for_all_fun(id_stats,nowTimeLinux))
        value=x.asDict()
        value['id_value_stats']=dict(zip(id_value_stats.keys(),map(lambda x:unicode(x),id_value_stats.values())))
        del value['id']
        del value['loan_date_unix']
        del value['id_value_time_cut']
        return Row(**value)

def map_time_id(x):
    d_list=[]
    id_value_time_cut={'15day':[],'365day':[],'180day':[],'90day':[],'60day':[],'30day':[],'30_90day':[],'90_180day':[],'180_365day':[]}
    if x.id_value:
        for l in x.id_value:
            check_num=0
            if l['eventtype']!='Loan':
                  check_num=1
            else:
                 key='Loan'+str(int(float(l['eventoccurtime']))/86400)+['partnercode']
                 if key not in d_list:
                     check_num=1
                     d_list.append(key)
            if check_num==1 and float(l['eventoccurtime']<=float(x.loan_date_unix)):
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 31536000:
                        id_value_time_cut['365day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 15552000:
                        id_value_time_cut['180day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 7776000:
                        id_value_time_cut['90day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 5184000:
                        id_value_time_cut['60day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 2592000:
                        id_value_time_cut['30day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventoccurtime']) < 1296000:
                        id_value_time_cut['15day'].append(l)
                    if float(x.loan_date_unix) - float(l['eventocurtime']) >= 2592000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 7776000):
                        id_value_time_cut['30_90day'].append(l)

                    if float(x.loan_date_unix) - float(l['eventocurtime']) >= 7776000 and ( float(x.loan_date_unix) - float(l['eventoccurtime']) < 15552000):
                        id_value_time_cut['90_180day'].append(l)

                    if float(x.loan_date_unix) - float(l['eventocurtime']) >= 15552000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 31536000):
                        id_value_time_cut['180_365day'].append(l)

    value=x.asDict()
    value['id_value_time_cut']=id_value_time_cut
    del value['id_value']
    return Row(**value)


 ##单跑手机号
def stats_fun_for_mobile_value(x):
    nowTimeLinux=float(x.loan_date_unix)
    mobile_stats={}
    for time_cut in set(time_cut_list):
        mobile_stats['all_all_'+time_cut]=x['mobile_value_time_cut'][time_cut]
        for event in set(['Loan','Register','Login','Trade','Apply']):
            mobile_stats[event+'_all_'+time_cut]=[x2 for x2 in mobile_stats['all_all_'+time_cut]if x2['eventtype']==event]
            for ind in set(ind_dict.keys()):
                mobile_stats[event+'_'+ind+'_'+time_cut]=[y2 for y2 in mobile_stats[event+'_all_'+time_cut]if y2['partnercode']in ind_dict[ind]]
            for ind in set(ind_dict.keys()):
                mobile_stats['all_'++ind+'_'+time_cut]=[y2 for y2 in mobile_stats['all_all_'+time_cut]if y2['partnercode']in ind_dict[ind]]
        mobile_value_stats={}
        if x.mobile:
            mobile_value_stats.update(stats_for_all_fun(mobile_stats,nowTimeLinux))
        value=x.asDict()
        value['mobile_value_stats']=dict(zip(mobile_value_stats.keys(),map(lambda x:unicode(x),mobile_value_stats.values())))
        del value['mobile']
        del value['loan_date_unix']
        del value['mobile_value_time_cut']
        return Row(**value)


def map_time_mobile(x):
    m_list=[]
    mobile_value_time_cut={'15day':[],'365day':[],'180day':[],'90day':[],'60day':[],'30day':[],'30_90day':[],'90_180day':[],'180_365day':[]}
    if x.mobile_value:
        for l in x.mobile_value:
            check_num=0
            if l['eventtype']!='Loan':
                check_num=1
            else:
                key='Loan'+str(int(float(l['eventoccurtime']))/86400+l['partnercode'])
                if key not in m_list:
                    check_num=1
                    m_list.append(key)
            if check_num==1 and float(l['eventoccurtime'])<=float(x.loan_date_unix):
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 31536000:
                    mobile_value_time_cut['365day'].append(l)
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 15552000:
                    mobile_value_time_cut['180day'].append(l)
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 7776000:
                    mobile_value_time_cut['90day'].append(l)
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 5184000:
                    mobile_value_time_cut['60day'].append(l)
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 2592000:
                    mobile_value_time_cut['30day'].append(l)
                if float(x.loan_date_unix) - float(l['eventoccurtime']) < 1296000:
                    mobile_value_time_cut['15day'].append(l)
                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 2592000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 7776000):
                    mobile_value_time_cut['30_90day'].append(l)
                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 7776000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 15552000):
                    mobile_value_time_cut['90_180day'].append(l)
                if float(x.loan_date_unix) - float(l['eventocurtime']) >= 15552000 and (float(x.loan_date_unix) - float(l['eventoccurtime']) < 31536000):
                    mobile_value_time_cut['180_365day'].append(l)
    value=x.asDict()
    value['mobile_value_time_cut']=mobile_value_time_cut
    del value['mobile_value']
    return Row(**value)


def match_var(Excelpath, Outputpath, filetype='both', existdate='True', backdate=''):
    try:
        dff = pd.read_excel(str(Excelpath))
    except:
        dff = pd.read_csv(str(Excelpath))
    print dff.columns()
    print u'请确认身份证、手机号、回溯时间是否为uid、umboile、date,是则输入1，否则输入为0'
    check_column = raw_input("")
    if str(check_column) == '0':
        id_name = raw_input('需要修改的身份证列名：')
        mobile_name = raw_input('需要修改的手机号列名：')
        date_name = raw_input('需要修改的回溯时间列名：')
        dff = dff.rename(columns={unicode(id_name): 'uid', unicode(mobile_name):'umobile', unicode(date_name):'date'})
        dff = dff.drop([x for x in dff.columns if re.search('Unnamed', x)], axis=1)
        for i in dff.columns:
            dff[i] = dff[i].astype(str)
        if existdate == 'True':
            dff['loan_date_unix'] = dff['date'].apply(lambda x: time.mktime(time.striptime(str(parse(x))[:19],'%Y-%m-%d %H:%M:%S')))
        else:
           dff['loan_date_unix'] = time.mktime(time.striptime(str(parse(backdate))[:19], '%Y-%m-%d  %H:%M:%S'))

        dff['umobile'] = dff['umobile'].apply(lambda x: x[:11])
        data = hc.createDataFrame(dff)
        d = hc.read.parquet('/user/sujun.cen/temp/icd')
        m = hc.read.parquet('/user/sujun.cen/temp/md')
        if filetype == 'both':
            df = data.join(d, data.uid == d.id, 'leftouter')
            df1 = df.join(m, df.umobile == m.mobile, 'leftouter')
            df2 = df1.rdd.map(map_time_all).map(stats_fun_for_all_value)
            df2.toDF().write.parquet(str(Outputpath), 'overwrite')
        elif filetype =='id':
            df = data.join(d, data.uid == d.id, 'leftouter')
            df1 = df.rdd.map(map_time_id).map(stats_fun_for_id_value)
            df1.toDF().write.parquet(str(Outputpath), 'overwrite')
        else:
            df = data.join(m, data.umobile == m.mobile, 'leftouter')
            df1 = df.rdd.map(map_time_mobile).map(stats_fun_for_mobile_value)
            df1.toDF().write.parquet(str(Outputpath), 'overwrite')
