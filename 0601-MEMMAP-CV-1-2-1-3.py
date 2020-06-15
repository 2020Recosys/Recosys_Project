#!/usr/bin/env python
# coding: utf-8
### Making CV MEMMAP files before modeling ###

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import itertools

### Load Data ###
온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

### Preprocessing ###
#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()


#각 clnt_id별로 shift로 1행씩 올림
구매여부.buy = 구매여부.buy.shift(-1)
g = 구매여부.groupby('clnt_id')

#1행씩 올리면 각 clnt_id별로 마지막 값은 다음 id의 값, 따라서 마지막 행 제거
구매여부.drop(g.tail(1).index, axis=0, inplace = True)
구매여부['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 구매여부.clnt_id, 구매여부.sess_id))

온라인2 = 온라인.copy()
온라인2.drop(['clnt_id', 'sess_id', 'trans_id', 'buy'], axis=1, inplace=True)

def to_flat(df):
    cc = df.groupby(['unique_id']).cumcount() + 1
    flat_df = df.set_index(['unique_id', cc]).unstack().sort_index(1, level=1)
    flat_df.columns = ['_'.join(map(str,i)) for i in flat_df.columns]
    flat_df.reset_index(inplace=True)
    return flat_df

온라인2 = to_flat(온라인2)
온라인2 = 온라인2.merge(구매여부, left_on='unique_id', right_on='unique_id')
온라인2.sort_values(by=['clnt_id','sess_id'], inplace=True)
print("flatten finished!")

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인2.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인2)]
idx = list(pd.Series(idx2) - pd.Series(idx1))

# session 당 구매 여부
X_resampled = 온라인2.iloc[:, 1:-3].fillna(0)
Y_resampled = 온라인2.fillna(0).buy

X_np = np.array(X_resampled)
Y_np = np.array(Y_resampled)
print("Ready for Memmap!")


### Memmap ###

from tempfile import mkdtemp
import os
import json

def make_path(file_name, directory='', is_make_temp_dir=False):
    """디렉토리와 파일명을 더해 경로를 만든다"""
    if is_make_temp_dir is True:
        directory = mkdtemp()
    if len(directory) >= 2 and not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory, file_name)

def make_memmap(mem_file_name, np_to_copy):
    """numpy.ndarray객체를 이용하여 numpy.memmap객체를 만든다"""
    memmap_configs = dict() # memmap config 저장할 dict
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape) # 형상 정보
    memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)   # dtype 정보
    json.dump(memmap_configs, open(mem_file_name+'.conf', 'w')) # 파일 저장
    # w+ mode: Create or overwrite existing file for reading and writing
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = np_to_copy[:]
    mm.flush() # memmap data flush
    return mm

def read_memmap(mem_file_name):
    """디스크에 저장된 numpy.memmap객체를 읽는다"""
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name+'.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', shape=tuple(memmap_configs['shape']), dtype=memmap_configs['dtype'])

# 파일기반 행렬 만들기
mem_file_name = make_path('mem_file_X.dat', directory='')
new_X_np        = make_memmap(mem_file_name, X_np)

mem_file_name = make_path('mem_file_Y.dat', directory='')
new_Y_np        = make_memmap(mem_file_name, Y_np)

print(type(X_np), type(new_X_np))
print(type(Y_np), type(new_Y_np))

del X_np, new_X_np, Y_np, new_Y_np

# 파일기반 행렬 읽기
mem_file_name = make_path('mem_file_X.dat', directory='')
new_X_np        = read_memmap(mem_file_name)
print("Memmap X finished!")

# 파일기반 행렬 읽기
mem_file_name = make_path('mem_file_Y.dat', directory='')
new_Y_np        = read_memmap(mem_file_name)
print("Memmap Y finished!")

# cross validation 생성
cv = StratifiedKFold(10, shuffle=True, random_state=42)
print('Made cv!')

idx = 0
print(idx)
for train_idx, test_idx in cv:
    X_train = new_X_np[train_idx,:]

    # 파일기반 행렬 만들기 - X_train
    fn = 'mem_file_X_train_' + idx + '.dat'
    mem_file_name = make_path(fn, directory='')
    new_X_train = make_memmap(mem_file_name, X_train)

    del X_train, new_X_train

    X_test = new_X_np[test_idx,:]

    # 파일기반 행렬 만들기 - X_test
    fn = 'mem_file_X_test_' + idx + '.dat'
    mem_file_name = make_path(fn, directory='')
    new_X_test = make_memmap(mem_file_name, X_test)

    del X_test, new_X_test

    y_train = new_Y_np[train_idx,:]

    # 파일기반 행렬 만들기 - y_train
    fn = 'mem_file_y_train_' + idx + '.dat'
    mem_file_name = make_path(fn, directory='')
    new_y_train = make_memmap(mem_file_name, y_train)

    del y_train, new_y_train

    y_test = new_Y_np[test_idx,:]

    # 파일기반 행렬 만들기 - y_test
    fn = 'mem_file_y_test_' + idx + '.dat'
    mem_file_name = make_path(fn, directory='')
    new_y_test = make_memmap(mem_file_name, y_test)

    del y_test, new_y_test

print("Memmap CV finished!")
