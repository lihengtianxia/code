#coding:utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame


#woe
woe_1=[0.61913676,0.174441007,-0.000867933,-0.193221447,-0.197503363,-0.203076642]
woe_2=[0.134529237,0.035033671,-0.088362663,-0.545764327]
woe_3=[0.325150156,-0.175826321,-0.191033387,-0.358397597]
woe_4=[-0.143678118,0.400871204]
woe_5=[0.71012215,0.635935842,0.244830252,-0.323424579,-0.407686419,-0.520665329]
woe_6=[0.345234607,-0.124550862,-0.447675055]
woe_7=[0.151052157,-0.132660011,-0.777518574]
woe_8=[0.393886424,-0.102790115,-0.669294998]
woe_9=[-0.592012448,-0.390732978,-0.303885388,-0.152831877,-0.000298171,0.218955442,0.462582955,0.539543996,0.701599855,1.269058821]
woe_10=[-0.161071479,0.320501999,0.385393316]
woe_11=[0.20682487,-0.312245285]
woe_12=[0.103761121,0.023583066,-0.624914859]
woe_13=[-0.128836761,0.503219669]
woe_14=[0.566432896,0.042751992,-0.084570274,-0.096033333,-0.325607774,-0.330144378]
woe_15=[-0.113939093,0.396710211,0.735806161]
woe_16=[0.194366548,0.025445047,-0.06137196,-0.54381873]
woe_17=[-0.118056038,0.092863059,0.845988345]
woe_18=[0.249103301,0.240872802,-0.075607561,-0.31996089]

coef = [-0.618789996, -0.519537384, -0.496929395, -0.513141626, -0.471811306, -0.87862784, -0.603887602,
            -0.621618978, -0.949852956, -0.831827702, -2.348014302, -0.732584225, -0.373614737, -0.710591315,
            -0.554327962, -1.006312642, -0.543769893, -0.843603958, -0.718976348
            ]
a = 410.4236223
b = 72.13475204
w0 = -2.348014302
s0 = a + b * w0
#x1
woe_m_ratio_lessthanA=0.61913676
woe_m_ratio_AtoB=0.174441007
woe_m_ratio_BtoC=-0.000867933
woe_m_ratio_CtoD=-0.193221447
woe_m_ratio_DtoE=-0.197503363
woe_m_ratio_morethanE=-0.203076642

#x2
woe_m_ratio_freq_lessthanA=0.134529237
woe_m_ratio_freq_AtoB=0.035033671
woe_m_ratio_freq_BtoC=-0.088362663
woe_m_ratio_freq_morethanC=-0.545764327

#x3
woe_i_ratio_freqlessthanA=0.325150156
woe_i_ratio_freqAtoB=-0.175826321
woe_i_ratio_freqBtoC=-0.191033387
woe_i_ratio_freqmorethanC=-0.358397597

#x4
woe_i_ratio_freq_weekday2lessthanA=-0.143678118
woe_i_ratio_freq_weekday2morethanA=0.400871204

#x5
woe_i_cnt_partner2lessthanA=0.71012215
woe_i_cnt_partner2AtoB=0.635935842
woe_i_cnt_partner2BtoC=0.244830252
woe_i_cnt_partner2CtoD=-0.323424579
woe_i_cnt_partner2DtoE=-0.407686419
woe_i_cnt_partner2morethanE=-0.520665329

#x6
woe_i_cnt_DebitCard2lessthanA=0.345234607
woe_i_cnt_DebitCard2AtoB=-0.124550862
woe_i_cnt_DebitCard2morethanB=-0.447675055


#x7
woe_i_freq_isroot2lessthanA=0.151052157
woe_i_freq_isroot2AtoB=-0.132660011
woe_i_freq_isroot2morethanB=-0.777518574


#x8
woe_i_freq_night2lessthanA=0.393886424
woe_i_freq_night2AtoB=-0.102790115
woe_i_freq_night2morethanB=-0.669294998


#x9
woe_i_province2lessthanA=-0.592012448
woe_i_province2AtoB=-0.390732978
woe_i_province2BtoC=-0.303885388
woe_i_province2CtoD=-0.152831877
woe_i_province2DtoE=-0.000298171
woe_i_province2EtoF=0.218955442
woe_i_province2FtoG=0.462582955
woe_i_province2GtoH=0.539543996
woe_i_province2HtoI=0.701599855
woe_i_province2morethanI=1.269058821


#x10
woe_i_cnt_trueipaddressprovince2lessthanA=-0.161071479
woe_i_cnt_trueipaddressprovince2AtoB=0.320501999
woe_i_cnt_trueipaddressprovince2morethanB=0.385393316


#x11
woe_m_cnt_indtwo2lessthanA=0.20682487
woe_m_cnt_indtwo2morethanA=-0.312245285

#x12
woe_i_max_freq_record_daily2lessthanA=0.103761121
woe_i_max_freq_record_daily2AtoB=0.023583066
woe_i_max_freq_record_daily2morethanB=-0.624914859


#x13
woe_i_ratio_freq_night2lessthanA=-0.128836761
woe_i_ratio_freq_night2morethanA=0.503219669

#x14
woe_i_max_freq_record_daily2_alllessthanA=0.566432896
woe_i_max_freq_record_daily2_allAtoB=0.042751992
woe_i_max_freq_record_daily2_allBtoC=-0.084570274
woe_i_max_freq_record_daily2_allCtoD=-0.096033333
woe_i_max_freq_record_daily2_allDtoE=-0.325607774
woe_i_max_freq_record_daily2_allmorethanE=-0.330144378

#x15
woe_m_freq_record2lessthanA=-0.113939093
woe_m_freq_record2AtoB=0.396710211
woe_m_freq_record2morethanB=0.735806161


#x16
woe_i_length_firstlessthanA=0.194366548
woe_i_length_firstAtoB=0.025445047
woe_i_length_firstBtoC=-0.06137196
woe_i_length_firstmorethanC=-0.54381873

#x17
woe_i_cnt_partner2_LoanlessthanA=-0.118056038
woe_i_cnt_partner2_LoanAtoB=0.092863059
woe_i_cnt_partner2_LoanmorethanB=0.845988345

#x18
woe_i_length_last2lessthanA=0.249103301
woe_i_length_last2AtoB=0.240872802
woe_i_length_last2BtoC=-0.075607561
woe_i_length_last2morethanC=-0.31996089

def getscore(i,x):
    score=round(b*coef[i]*x,7)
    return score

df=pd.read_csv("c:/bin_train_modified.csv")
samples,feat=df.shape

tmp_x1=[0]*samples
tmp_x2=[0]*samples
tmp_x3=[0]*samples
tmp_x4=[0]*samples
tmp_x5=[0]*samples
tmp_x6=[0]*samples
tmp_x7=[0]*samples
tmp_x8=[0]*samples
tmp_x9=[0]*samples
tmp_x10=[0]*samples
tmp_x11=[0]*samples
tmp_x12=[0]*samples
tmp_x13=[0]*samples
tmp_x14=[0]*samples
tmp_x15=[0]*samples
tmp_x16=[0]*samples
tmp_x17=[0]*samples
tmp_x18=[0]*samples

feature=['m_ratio_freq_record2_all_P2pweb_90day','m_ratio_freq_night2_Login_P2pweb_30_90day','i_ratio_freq_weekend2_Loan_con_15day','i_ratio_freq_weekday2_Loan_Inconsumerfinance_30day','i_cnt_partner2_Loan_Imbank_15day','i_cnt_DebitCard2_all_all_365day','i_freq_isroot2_all_all_30day','i_freq_night2_Loan_all_15day','i_province2_asD','i_cnt_trueipaddressprovince2_all_all_180_365day','m_cnt_indtwo2_Register_all_180day','i_max_freq_record_daily2_Login_P2pweb_15day','i_ratio_freq_night2_all_Inconsumerfinance_180day','i_max_freq_record_daily2_all_Unconsumerfinance_365day','m_freq_record2_Loan_Inconsumerfinance_90_180day','i_length_first_last2_all_Consumerfinance_90day','i_cnt_partner2_Loan_Inconsumerfinance_90day','i_length_last2_Loan_Bank_365day']
# tmp=[[0]*18]*samples
for i in range(samples):
    #x1
    if(df.ix[i,feature[0]]<=0.361):
        tmp_x1[i]=woe_m_ratio_lessthanA
    elif(df.ix[i,feature[0]]<=0.462):
        tmp_x1[i] = woe_m_ratio_AtoB
    elif(df.ix[i,feature[0]]<=0.539):
        tmp_x1[i] = woe_m_ratio_BtoC
    elif (df.ix[i,feature[0]] <= 0.615):
        tmp_x1[i] = woe_m_ratio_CtoD
    elif (df.ix[i,feature[0]] <= 0.709):
        tmp_x1[i] = woe_m_ratio_DtoE
    else:
        tmp_x1[i] = woe_m_ratio_morethanE
     #x2
    if (df.ix[i,feature[1]] <= 0):
        tmp_x2[i] = woe_m_ratio_freq_lessthanA
    elif (df.ix[i,feature[1]] <= 0.125):
        tmp_x2[i] = woe_m_ratio_freq_AtoB
    elif (df.ix[i,feature[1]] <= 0.333):
        tmp_x2[i] = woe_m_ratio_freq_BtoC
    else:
        tmp_x2[i] = woe_m_ratio_freq_morethanC
    #x3
    if (df.ix[i,feature[2]] <= 0):
        tmp_x3[i] = woe_i_ratio_freqlessthanA
    elif (df.ix[i,feature[2]] <= 0.2):
        tmp_x3[i] = woe_i_ratio_freqAtoB
    elif (df.ix[i,feature[2]] <= 0.366):
        tmp_x3[i] = woe_i_ratio_freqBtoC
    else:
        tmp_x3[i] = woe_i_ratio_freqmorethanC
    #x4
    if (df.ix[i,feature[3]] <= 0):
        tmp_x4[i] = woe_i_ratio_freq_weekday2lessthanA
    else:
        tmp_x4[i] = woe_i_ratio_freq_weekday2morethanA
    #x5
    if (df.ix[i,feature[4]] <= 3):
        tmp_x5[i] = woe_i_cnt_partner2lessthanA
    elif (df.ix[i,feature[4]] <= 5):
        tmp_x5[i] = woe_i_cnt_partner2AtoB
    elif (df.ix[i,feature[4]] <= 7):
        tmp_x5[i] = woe_i_cnt_partner2BtoC
    elif (df.ix[i,feature[4]] <= 9):
        tmp_x5[i] = woe_i_cnt_partner2CtoD
    elif (df.ix[i,feature[4]] <= 13):
        tmp_x5[i] = woe_i_cnt_partner2DtoE
    else:
        tmp_x5[i] = woe_i_cnt_partner2morethanE
    #x6
    if (df.ix[i,feature[5]] <= 1):
        tmp_x6[i] = woe_i_cnt_DebitCard2lessthanA
    elif (df.ix[i,feature[5]] <= 2):
        tmp_x6[i] = woe_i_cnt_DebitCard2AtoB
    else:
        tmp_x6[i] = woe_i_cnt_DebitCard2morethanB

   #x7
    if (df.ix[i,feature[6]] <= 1):
        tmp_x7[i] = woe_i_freq_isroot2lessthanA
    elif (df.ix[i,feature[6]] <= 2):
        tmp_x7[i] = woe_i_freq_isroot2AtoB
    else:
        tmp_x7[i] = woe_i_freq_isroot2morethanB

    #x8
    if (df.ix[i,feature[7]] <= 1):
        tmp_x8[i] = woe_i_freq_night2lessthanA
    elif (df.ix[i,feature[7]] <= 2):
        tmp_x8[i] = woe_i_freq_night2AtoB
    else:
        tmp_x8[i] = woe_i_freq_night2morethanB

    #x9
    if (df.ix[i,feature[8]] =='LN'):
        tmp_x9[i] = woe_i_province2lessthanA
    elif (df.ix[i,feature[8]] in ['IM','SX']):
        tmp_x9[i] = woe_i_province2AtoB
    elif (df.ix[i,feature[8]] in ['SN','CQ']):
        tmp_x9[i] = woe_i_province2BtoC
    elif (df.ix[i,feature[8]] in ['HA','GS','SC','HN']):
        tmp_x9[i] = woe_i_province2CtoD
    elif (df.ix[i,feature[8]] in ['HB','GX','HI','SD','TJ','HL','GD']):
        tmp_x9[i] = woe_i_province2DtoE
    elif (df.ix[i,feature[8]] in ['FJ','HE','JS','JL','JX','ZJ']):
        tmp_x9[i] = woe_i_province2EtoF
    elif (df.ix[i,feature[8]] in ['GZ','AH']):
        tmp_x9[i] = woe_i_province2FtoG
    elif (df.ix[i,feature[8]] in ['YN']):
        tmp_x9[i] = woe_i_province2GtoH
    elif (df.ix[i,feature[8]] in ['XJ','NX']):
        tmp_x9[i] = woe_i_province2HtoI
    elif (df.ix[i,feature[8]] in ['BJ','QH','SH']):
        tmp_x9[i] = woe_i_province2morethanI
    #x10
    if (df.ix[i,feature[9]] <= 1):
        tmp_x10[i] = woe_i_cnt_trueipaddressprovince2lessthanA
    elif (df.ix[i,feature[9]] <= 2):
        tmp_x10[i] = woe_i_cnt_trueipaddressprovince2AtoB
    else:
        tmp_x10[i] = woe_i_cnt_trueipaddressprovince2morethanB

    #x11
    if (df.ix[i,feature[10]] <= 2):
        tmp_x11[i] = woe_m_cnt_indtwo2lessthanA
    else :
        tmp_x11[i] = woe_m_cnt_indtwo2morethanA
    #x12
    if (df.ix[i,feature[11]] <= 1):
        tmp_x12[i] = woe_i_max_freq_record_daily2lessthanA
    elif (df.ix[i,feature[11]] <= 3):
        tmp_x12[i] = woe_i_max_freq_record_daily2AtoB
    else:
        tmp_x12[i] = woe_i_max_freq_record_daily2morethanB

    #x13
    if (df.ix[i,feature[12]] <= 0):
        tmp_x13[i] = woe_i_ratio_freq_night2lessthanA
    else :
        tmp_x13[i] = woe_i_ratio_freq_night2morethanA

    #x14
    if (df.ix[i,feature[13]] <= 1):
        tmp_x14[i] = woe_i_max_freq_record_daily2_alllessthanA
    elif (df.ix[i,feature[13]] <= 2):
        tmp_x14[i] = woe_i_max_freq_record_daily2_allAtoB
    elif (df.ix[i,feature[13]] <= 3):
        tmp_x14[i] = woe_i_max_freq_record_daily2_allBtoC
    elif (df.ix[i,feature[13]] <= 4):
        tmp_x14[i] = woe_i_max_freq_record_daily2_allCtoD
    elif (df.ix[i,feature[13]] <= 5):
        tmp_x14[i] = woe_i_max_freq_record_daily2_allDtoE
    else:
        tmp_x14[i] = woe_i_max_freq_record_daily2_allmorethanE
    #x15
    if (df.ix[i,feature[14]] <= 1):
        tmp_x15[i] = woe_m_freq_record2lessthanA
    elif (df.ix[i,feature[14]] <= 2):
        tmp_x15[i] = woe_m_freq_record2AtoB
    else :
        tmp_x15[i] = woe_m_freq_record2morethanB
    #x16
    if (df.ix[i,feature[15]] <= 8.058):
        tmp_x16[i] = woe_i_length_firstlessthanA
    elif (df.ix[i,feature[15]] <= 32.582):
        tmp_x16[i] = woe_i_length_firstAtoB
    elif (df.ix[i,feature[15]]  <= 56.976):
        tmp_x16[i] = woe_i_length_firstBtoC
    else :
        tmp_x16[i] = woe_i_length_firstmorethanC

    #x17
    if (df.ix[i,feature[16]] <= 1):
        tmp_x17[i] = woe_i_cnt_partner2_LoanlessthanA
    elif (df.ix[i,feature[16]] <= 2):
        tmp_x17[i] = woe_i_cnt_partner2_LoanAtoB
    else :
        tmp_x17[i] = woe_i_cnt_partner2_LoanmorethanB

    #x18
    if (df.ix[i,feature[17]] <= 1.845):
        tmp_x18[i] = woe_i_length_last2lessthanA
    elif (df.ix[i,feature[17]] <= 17.77):
        tmp_x18[i] = woe_i_length_last2AtoB
    elif (df.ix[i,feature[17]] <= 50.92):
        tmp_x18[i] = woe_i_length_last2BtoC
    else:
        tmp_x18[i] = woe_i_length_last2morethanC
woe=np.transpose([tmp_x1,tmp_x2,tmp_x3,tmp_x4,tmp_x5,tmp_x6,tmp_x7,tmp_x8,tmp_x9,tmp_x10,tmp_x11,tmp_x12,tmp_x13,tmp_x14,tmp_x15,tmp_x16,tmp_x17,tmp_x18])
woe_com=DataFrame(woe,columns=feature)
N,M=woe_com.shape
s=[s0]*N

for i in range(N):
    for j in range(M):
        x=woe_com.iat[i,j]
        s[i]=s[i]+getscore(j,x)
woe_com['score']=s
# woe_com.insert(M,'score','i_length_last2_Loan_Bank_365day')
woe_com.to_csv('lbd_last_score.csv')

