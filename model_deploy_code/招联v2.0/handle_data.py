#-*- coding:utf8 -*-
#### 1.所有编码格式统一为utf-8
__author__ = "fraudmetrix-chaoqin"

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
import simplejson as json
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

#### 2.获取文件路径
filepath = os.path.split(os.path.realpath(__file__))[0]

def numeric_binning(s, val):
    if (s==u"-999" or s==-999) and val==-999:
        return True
    if (s==u"-1" or s==-1) and val==-1:
        return True
    if (s==u"-1" or s==-1) and val!=-1:
        return False
    s=s.strip(" ")
    _ele=[]
    _ele.append(s[0])
    _ele.append(s[len(s)-1])
    flag=0
    val=round(val, 3)
    if s.find('+')>=0:
        _ele.append(float(s[1:len(s)-1].split(", ")[0]))
        if _ele[0]=="(" and _ele[2]<val:
            flag+=2
        elif _ele[0]=="[" and _ele[2]<=val:
            flag+=1
    elif s.find('+')<0:
        _ele.extend([float(x) for x in s[1:len(s)-1].split(", ")])
        if _ele[0]=="(" and _ele[2]<val:
            flag+=1
        elif _ele[0]=="[" and _ele[2]<=val:
            flag+=1
        if _ele[1]==")" and _ele[3]>val:
            flag+=1
        elif _ele[1]=="]" and _ele[3]>=val:
            flag+=1
    if flag==2:
        return True
    else:
        return False

def categorical_binning(s, val):
    s=s.strip(" ")
    if s==val:
        return True
    else:
        return False

def binning(df_,val):
    import copy
    df=copy.copy(df_)
    var_name=df.iloc[0].tolist()
    df.index=[x for x in range(df.shape[0])]
    df=df.drop(0)
    df.columns=var_name
    label_=0
    range_=''
    score_=0
    if type(val)==type(''):
        # 判断在哪个bin
        for i in range(len(df)):
            if categorical_binning(df.iloc[i][1], val)==True:
                 # 输出label，range和score
                label_=df.iloc[i][0]
                range_=df.iloc[i][1]
                score_=df.iloc[i][2]
                return label_, range_, score_
    elif type(val)==type(1.0) or type(val)==type(1):
        for i in range(len(df)):
            if numeric_binning(df.iloc[i][1], val)==True:
                # 输出label，range和score
                label_=df.iloc[i][0]
                range_=df.iloc[i][1]
                score_=df.iloc[i][2]
                return label_, range_, score_
    return label_, range_, score_


### 3.返回值必须为dict格式
def handle(params):
    df_all = pd.read_excel(filepath+"/zhaolian_bin.xlsx", header=None)
    result={"model_score": 0}
    for key in params.keys():
        # feidai_bin第三列作为分
        df_=df_all[df_all[3]==key.upper()]
        # print df_
        val=params.get(key)
        # print key
        label, range, score=binning(df_, val)
        # 输出模型分和label，如果不需要模型分就把下面这行注释掉
        result["model_score"]=score+result["model_score"]
        result["l_"+key.lower()]=label
    return result

