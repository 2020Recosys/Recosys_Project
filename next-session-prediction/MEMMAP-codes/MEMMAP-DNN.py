#!/usr/bin/env python
# coding: utf-8
### 1-2 DNN ###

import numpy as np
from tempfile import mkdtemp
import os
import json
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Masking
from keras.optimizers import RMSprop
from keras import backend as K
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
print('Library loaded')

### functions for reading memmap ###

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
        return np.memmap(mem_file_name, mode='r+', shape=tuple(memmap_configs['shape']), dtype=memmap_configs['dtype'])

### Metrics ###

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

### Modeling ###

def one_cv_for_one_algo(algorithm, cv_idx):

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

def dnn_models():
    dnn_model = Sequential()
    dnn_model.add(Dense(32, activation='relu', input_shape=(9639,)))
    dnn_model.add(Dense(16, activation='relu'))
    dnn_model.add(Dense(1, activation='sigmoid'))
    dnn_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc', f1_m, precision_m, recall_m])
    return dnn_model

all_scores = []
for cv_idx in tqdm(range(0, 10)):
    print(cv_idx)

    algorithm = KerasClassifier(build_fn=dnn_models, epochs=25, batch_size=1000, verbose=1)
    scores = one_cv_for_one_algo(algorithm, cv_idx)

    print(str(cv_idx),'번 cv: ',scores)
    all_scores.append(scores)

import pickle
with open('DNN_1-2-1-3_scores.pkl','wb') as f:
    pickle.dump(all_scores, f)
