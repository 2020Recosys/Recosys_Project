#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[3]:


온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)


# In[4]:


온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))


# In[5]:


#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()


# In[6]:


#각 clnt_id별로 shift로 1행씩 올림
구매여부.buy = 구매여부.buy.shift(-1)
g = 구매여부.groupby('clnt_id')
#1행씩 올리면 각 clnt_id별로 마지막 값은 다음 id의 값, 따라서 마지막 행 제거
구매여부.drop(g.tail(1).index, axis=0, inplace = True)
구매여부['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 구매여부.clnt_id, 구매여부.sess_id))


# In[7]:


온라인2 = 온라인.copy()
온라인2.drop(['clnt_id', 'sess_id', 'trans_id', 'buy'], axis=1, inplace=True)


# In[8]:


def to_flat(df):
    cc = df.groupby(['unique_id']).cumcount() + 1
    flat_df = df.set_index(['unique_id', cc]).unstack().sort_index(1, level=1)
    flat_df.columns = ['_'.join(map(str,i)) for i in flat_df.columns]
    flat_df.reset_index(inplace=True)
    return flat_df


# In[9]:


온라인2 = to_flat(온라인2)
온라인2 = 온라인2.merge(구매여부, left_on='unique_id', right_on='unique_id')
온라인2.sort_values(by=['clnt_id','sess_id'], inplace=True)


# In[ ]:


온라인2.to_csv('./온라인_전처리_flat_1-2.csv', encoding='utf-8', index=False)


# In[ ]:


# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인2.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인2)]
idx = list(pd.Series(idx2) - pd.Series(idx1))


# In[ ]:


# session 당 구매 여부
X_resampled = 온라인2.iloc[:, 1:-3].fillna(0)
Y_resampled = 온라인2.fillna(0).buy


# In[ ]:


X_resampled.to_csv('./X_resampled_1-2.csv', encoding='utf-8', index=False)
Y_resampled.to_csv('./Y_resampled_1-2.csv', encoding='utf-8', index=False)

