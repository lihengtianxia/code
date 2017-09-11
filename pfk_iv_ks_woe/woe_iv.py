#coding:utf-8

#分bin
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
df=pd.read_csv('d:/bin_train_modified.csv')
target='Disbursed'
feature=[x for x in df.columns if x not in [target]]
train_x,test_x,train_y,test_y=train_test_split(df[feature],df[target],test_size=0.3,random_state=0)
first_line=df.columns
for ii in range(len(first_line)):
    point=[1]*11
    for i in range(10):
        point[i] = np.percentile(df[first_line[ii]], i*10)
    point[10]=point[10]+float("inf")
    new_point=list(set(point))
    new_point=sorted(new_point)
    df["B_"+first_line[ii]]=pd.cut(df[first_line[ii]],new_point,include_lowest=True,right=False)
df.to_csv("bin_train_modified.csv")


import pandas as pd
import  numpy as np
import math
from math import e
df=pd.read_csv('d:/train_modified_1.csv')
df=pd.read_csv('d:/bin_train_modified.csv')
totalgood=df['Disbursed'].value_counts()[0]
totalbad=df.shape[0]-totalgood
#woe
def calculate_WOE(x,p,q):

   Good=df.loc[(df['Existing_EMI'] >p) & (df['Existing_EMI'] <=q) & (df['Disbursed'] == 0), ['Existing_EMI']].count()
   Bad=df.loc[(df['Existing_EMI'] >p) & (df['Existing_EMI'] <=q) & (df['Disbursed'] == 1), ['Existing_EMI']].count()
   woe=math.log((Bad/totalbad)/(Good/totalgood),e)
   return woe

#calculate iv
def calculate_iv(df):
   totalgood = df['Disbursed'].value_counts()[0]
   totalbad = df.shape[0] - totalgood
   inside_good = dict()
   inside_bad = dict()
   iv = 0
   cats = df['Existing_EMI'].value_counts().index
   # cats=list(set(cats))
   # newcats=sorted(cats)
   for c in cats:
      EMI_df = df[df['Existing_EMI'] == c]
      # inside_total = EMI_df['Existing_EMI'].count()
      good = EMI_df.loc[(EMI_df['Disbursed'] == 0), ['Existing_EMI']].count()
      bad = EMI_df.loc[(EMI_df['Disbursed'] == 1), ['Existing_EMI']].count()
      inside_good[c] = good
      inside_bad[c] = bad
      woe=math.log((bad/totalbad)/(good/totalgood),e)
      iv+=(bad/totalbad-good/totalgood)*woe
   return iv
#ks
# def calc_ks_qcut(df, response='target', score='score', good=0, bad=1, name='score_band'):
#     # import copy
#     # df=copy.copy(df)
#     import numpy as np
#     df=df[[response,score]]
#     df=df.sort_values(score)
#     df[name]=pd.qcut(df[df[score]>0][score],10)
#     ks_table=pd.crosstab(df[name], df[response])
#     ks_table['cum_good_pct']=ks_table[good].cumsum()/ks_table[good].sum()
#     ks_table['cum_bad_pct']=ks_table[bad].cumsum()/ks_table[bad].sum()
#     ks_table['ks']=np.abs(ks_table['cum_good_pct']-ks_table['cum_bad_pct'])
#     return ks_table
#ks
def calculate_ks(df):
   totalgood = df['Disbursed'].value_counts()[0]
   totalbad = df.shape[0] - totalgood
   cats = df['Existing_EMI'].value_counts().index
   cum_good=dict()
   cum_bad=dict()
   cum_good_percent=dict()
   cum_bad_percent=dict()
   ks=dict()
   for c in cats:
      cum_good[c]=df.loc[(df['Existing_EMI'] == c)&(df['Disbursed'] == 0),['Existing_EMI']].cumsum()
      cum_bad[c] = df.loc[(df['Existing_EMI'] == c) & (df['Disbursed'] == 1), ['Existing_EMI']].cumsum()
      cum_good_percent[c]=cum_good[c]/totalgood
      cum_bad_percent[c]=cum_bad[c]/totalbad
      ks[c]=np.abs(cum_bad_percent[c]-cum_good_percent[c])
   return ks.values().max()


#psi

def calculate_psi(df):
   total_train=train_x.shpae[0]
   total_test=test_x.shape[0]
   cats = total_train['Existing_EMI'].value_counts().index

   train_sum=dict()
   test_sum=dict()
   psi=dict()
   s=0
   for c in cats:
      train_sum[c]=total_train.loc[(df['Existing_EMI'] == c),['Existing_EMI']].count()
      test_sum[c] = total_test.loc[(df['Existing_EMI'] == c) , ['Existing_EMI']].count()
      psi[c]=np.abs((train_sum[c]/total_train-test_sum[c]/total_test)*math.log((train_sum[c]/total_train)/(test_sum[c]/total_test),e))
      s=s+psi[c]
   return s


