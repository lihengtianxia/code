#coding:utf-8
import pandas as pd

#简单的、可复用的方法，轻松为任意变量分箱
def binning(col, cut_points, labels=None):

  #Define min and max values:

  minval = col.min()

  maxval = col.max()

  #利用最大值和最小值创建分箱点的列表

  break_points = [minval] + cut_points + [maxval]

  #如果没有标签，则使用默认标签0 ... (n-1)

  if not labels:

    labels = range(len(cut_points)+1)

  #使用pandas的cut功能分箱

  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)

  return colBin


# Monotonic Binning with Python,最优分箱
import pandas as pd
import numpy as np
import scipy.stats.stats as stats

# import data
data = pd.read_csv("/home/liuwensui/Documents/data/accepts.csv", sep=",", header=0)


# define a binning function
def mono_bin(Y, X, n=20):


# fill missings with median
X2 = X.fillna(np.median(X))
r = 0
while np.abs(r) < 1:
    d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
d2 = d1.groupby('Bucket', as_index=True)
r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
n = n - 1
d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
d3['max_' + X.name] = d2.max().X
d3[Y.name] = d2.sum().Y
d3['total'] = d2.count().Y
d3[Y.name + '_rate'] = d2.mean().Y
d4 = (d3.sort_index(by='min_' + X.name)).reset_index(drop=True)
print "=" * 60
print d4
mono_bin(data.bad, data.ltv)
mono_bin(data.bad, data.bureau_score)
mono_bin(data.bad, data.age_oldest_tr)
mono_bin(data.bad, data.tot_tr)
mono_bin(data.bad, data.tot_income)


#ks计算
def calc_ks_qcut(df, response='target', score='score', good=0, bad=1, name='score_band'):
    # import copy
    # df=copy.copy(df)
    import numpy as np
    df=df[[response,score]]
    df=df.sort_values(score)
    df[name]=pd.qcut(df[df[score]>0][score],10)
    ks_table=pd.crosstab(df[name], df[response])
    ks_table['cum_good_pct']=ks_table[good].cumsum()/ks_table[good].sum()
    ks_table['cum_bad_pct']=ks_table[bad].cumsum()/ks_table[bad].sum()
    ks_table['ks']=np.abs(ks_table['cum_good_pct']-ks_table['cum_bad_pct'])
    return ks_table


#woe

#iv，woe
import pandas as pd
import random
import numpy as np

# 读取数据
df1 = pd.read_csv('target.csv')
df2 = pd.read_csv('non_target.csv')
variable_list = ['', '', '', '', '']
# notarget随机抽取80000条记录
random.seed(10)
neg = df2.copy().ix[random.sample(list(df2.index), 80000)]
pos = df1.copy()
# 选取正（pos)负(neg)样本中flag=1的记录,选出variable_list中的变量，并合并正负样本
neg = neg[neg.flag == 1][variable_list]
pos = pos[pos.flag == 1][variable_list]
neg['target'] = 0
pos['target'] = 1
data = pd.concat([neg, pos])
# print记录条数
print (neg.shape, pos.shape, data.shape)

###

raw_data = data.copy()
for i in raw_data[variable_list].columns:
    column_name = i + '_cat'
    bincut = raw_data[i].dropna().quantile([i / 10.0 for i in range(10)]).drop_duplicates().values
    raw_data[column_name] = 'missing'
    for j in range(len(bincut) - 1):
        raw_data[column_name][(raw_data[i] == bincut[j])]=str(j) + ":[" + str(bincut[j]) + "," + str(bincut[j + 1]) + ")"
        raw_data[column_name][raw_data[i] >= bincut[len(bincut) - 1]] = str(j + 1) + ":>=" + str(
            bincut[len(bincut) - 1])
        bin_data = raw_data.iloc[:, data.shape[1] - 1:raw_data.shape[1]]
fun = ['count', 'sum', 'mean']
result = bin_data['target'].groupby([bin_data[bin_data.columns[1]]]).agg(fun)
result['variable_name'] = bin_data.columns[1]
for k in bin_data.columns[2:]:
    profile = bin_data['target'].groupby([bin_data[k]]).agg(fun)
profile['variable_name'] = k
result = pd.concat([result, profile])
result = pd.DataFrame(result[['variable_name', 'count', 'sum', 'mean']])
percent = pd.DataFrame(result['count'] / data.shape[0])
profile_final = pd.concat([result, percent], axis=1)
profile_final.columns = ['variable_name', '#Records', '#Target', 'Resp_Rate', 'Percent']

profile_final.to_csv('profile_final.csv')

woe = np.log((profile_final['#Target'] / bin_data['target'].sum()) / (
(profile_final['#Records'] - profile_final['#Target']) / (data.shape[0] - bin_data['target'].sum())))
iv_value = ((profile_final['#Target'] / bin_data['target'].sum()) - (
profile_final['#Records'] - profile_final['#Target']) / (data.shape[0] - bin_data['target'].sum())) * woe
iv = pd.concat([profile_final, pd.DataFrame(iv_value)], axis=1)
iv.columns = ['variable_name', '#Records', '#Target', 'Resp_Rate', 'Percent', 'iv']
result_iv = pd.DataFrame(iv['iv'].groupby(iv['variable_name']).sum())
result_iv.columns = ['variable_name', 'iv']
result_iv = result_iv.sort_values(by='iv', ascending=False)
result_iv.to_csv('result_iv.csv')

#psi



if __name__ =='__main__':
    # 为年龄分箱:



