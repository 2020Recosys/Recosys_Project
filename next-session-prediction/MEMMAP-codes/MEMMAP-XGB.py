#!/usr/bin/env python
# coding: utf-8

# # Data Load 

# In[ ]:


import numpy as np
from tempfile import mkdtemp
import os
import json
from tqdm.notebook import tqdm


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


# In[ ]:


def make_path(file_name, directory='', is_make_temp_dir=False):
    """디렉토리와 파일명을 더해 경로를 만든다"""
    if is_make_temp_dir is True:
        directory = mkdtemp()
    if len(directory) >= 2 and not os.path.exists(directory):
        os.makedirs(directory)    
    return os.path.join(directory, file_name)

def read_memmap(mem_file_name):
    """디스크에 저장된 numpy.memmap객체를 읽는다"""
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name+'.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+',                          shape=tuple(memmap_configs['shape']),                          dtype=memmap_configs['dtype'])


# # Modeling

# In[ ]:


def one_cv_for_one_algo(algorithm, X_train, y_train):
    clf = make_pipeline(SMOTE(random_state=0), algorithm)
    clf.fit(X_train, y_train)
    del X_train, y_train
    
    # For Testing
    # X_test
    fn = 'mem_file_X_test_' + str(cv_idx) + '.dat'
    mem_file_name = make_path(fn, directory='')
    X_test = read_memmap(mem_file_name)
    print('X_test loaded')

    # y_test
    fn = 'mem_file_y_test_' + str(cv_idx) + '.dat'
    mem_file_name = make_path(fn, directory='')
    y_test = read_memmap(mem_file_name)
    print('y_test loaded')
    
    y_pred = clf.predict(X_test)
    del X_test
    
    s1 = accuracy_score(y_true=y_test, y_pred=y_pred)
    s2 = precision_score(y_true=y_test, y_pred=y_pred)
    s3 = recall_score(y_true=y_test, y_pred=y_pred)
    s4 = f1_score(y_true=y_test, y_pred=y_pred)
    del y_test
    
    print('accuracy:', s1)
    print('precision:', s2)
    print('recall:', s3)
    print('f1:', s4)
    return [s1, s2, s3, s4]


# In[ ]:


all_scores = []
for cv_idx in tqdm(range(0, 10)):
    print(cv_idx)
    
    ## For Training
    # X_train
    fn = 'mem_file_X_train_' + str(cv_idx) + '.dat'
    mem_file_name = make_path(fn, directory='')
    X_train = read_memmap(mem_file_name)
    print('X_train loaded')

    # y_train
    fn = 'mem_file_y_train_' + str(cv_idx) + '.dat'
    mem_file_name = make_path(fn, directory='')
    y_train = read_memmap(mem_file_name)
    print('y_train loaded')
    
    # XGB with GPU:0
    algorithm = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0,
                                  learning_rate = 0.05, n_estimators=300,
                                  max_depth=3, verbosity=2, random_state=0)
    scores = one_cv_for_one_algo(algorithm, X_train, y_train)

    print(str(cv_idx),'번 cv: ',scores)
    all_scores.append(scores)    


# In[ ]:


import pickle
with open('XGB_1-2-1-3_scores.pkl','w') as f:
    pickle.dump(all_scores, f)

