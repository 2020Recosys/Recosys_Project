## 1-2. 현재 세션(1개)의 모든 클릭 로그를 대상으로 MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을 
# 사용해서 구매 예측  [예린]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

온라인 = pd.read_csv('C:/Users/JKKIM/Desktop/Recommender/온라인_전처리_final_32columns/온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()

구매여부1 = pd.DataFrame()
for id in tqdm_notebook(구매여부['clnt_id'].unique()):
    temp = 구매여부[구매여부['clnt_id'] == id]
    temp.buy = temp.buy.shift(-1)
    temp = temp.dropna(axis = 0)
    구매여부1 = pd.concat([구매여부1, temp])

온라인2 = 온라인.copy()
온라인2.drop(['clnt_id', 'sess_id', 'trans_id', 'buy'], axis=1, inplace=True)

def to_flat(df):
    cc = df.groupby(['unique_id']).cumcount() + 1
    flat_df = df.set_index(['unique_id', cc]).unstack().sort_index(1, level=1)
    flat_df.columns = ['_'.join(map(str,i)) for i in flat_df.columns]
    flat_df.reset_index(inplace=True)
    return flat_df

온라인2 = to_flat(온라인2)
#온라인 = pd.merge(온라인, 구매여부1, left_on = ['clnt_id', 'sess_id'], right_on = ['clnt_id', 'sess_id'])
온라인2 = 온라인2.merge(구매여부1, left_on='unique_id', right_on='unique_id')
온라인2.sort_values(by=['clnt_id','sess_id'], inplace=True)
features = 온라인2.columns[1:-3]

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Masking
from keras.optimizers import RMSprop
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras import backend as K
'''
import os

import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

KTF.set_session(session)
'''


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

def make_padding_and_oversample(X, Y, length=350):
    max_len = length #np.percentile(pd.Series(idx),99)
    X_padding = sequence.pad_sequences(X, maxlen=max_len, padding='pre', truncating='post')
    X_padding2 = X_padding.reshape(X.shape[0], max_len* X_padding.shape[2])
    print("pad_sequences 완료")

    smote = SMOTE(random_state=0)
    X_resampled, Y_resampled = smote.fit_resample(X_padding2, Y)
    print("smote 완료")
    X_resampled = X_resampled.reshape(X_resampled.shape[0], max_len, X_padding.shape[2])
    return X_padding, X_resampled, Y_resampled

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인2.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인2)]

'''
# 구매 시점이 있으면 index 저장하고 없으면 마지막 index 그대로 가져옴
idx3 = []
for i, j in tqdm_notebook(zip(idx1, idx2), total=len(idx1)):
    temp = 온라인.buy.iloc[i:j]
    try:
        idx3.append(temp[temp == 1].index[0])
    except:
        idx3.append(j)
'''


idx = list(pd.Series(idx2) - pd.Series(idx1))
max(idx), np.percentile(pd.Series(idx),99)

온라인_x = 온라인2.iloc[:, 1:-3]
온라인_x = np.array(온라인_x)

# session 당 구매 여부
온라인_y = 온라인2.buy

X_padded, X_resampled, Y_resampled = make_padding_and_oversample(온라인_x, 온라인_y, length=max(idx))

def dnn_models():
    dnn_model = Sequential()
    dnn_model.add(Dense(32, activation='relu', input_shape=(X_resampled.shape[1],)))
    dnn_model.add(Dense(16, activation='relu'))
    dnn_model.add(Dense(1, activation='sigmoid'))
    dnn_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc', f1_m, precision_m, recall_m])
    return dnn_model

## Cross-validation
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier

#X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
cv = StratifiedKFold(10, shuffle=True, random_state=42)

clf = GaussianNB()
acc_scores1 = cross_validate(clf, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf2 = DecisionTreeClassifier(random_state=0)
acc_scores2 = cross_validate(clf2, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf3 = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=3, verbosity=2, random_state=0)
acc_scores3 = cross_validate(clf3, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf4 = LogisticRegression(max_iter=1000, random_state=0)
acc_scores4 = cross_validate(clf4, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf5 = LinearSVC(random_state=0)
acc_scores5 = cross_validate(clf5, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf6 = KerasClassifier(build_fn=dnn_models, epochs=25, batch_size=1000, verbose=1)
acc_scores6 = cross_validate(clf6, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])




## 1-3. 현재 세션 앞 부분의 1~10개의 클릭 로그를 대상으로 구매 예측을 할 때, MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, 
# Logistic Regression, Linear SVM을 사용해서 구매 예측  [예린]
def dnn_models1():
    dnn_model = Sequential()
    dnn_model.add(Dense(32, activation='relu', input_shape=(X_resampled1.shape[1],)))
    dnn_model.add(Dense(16, activation='relu'))
    dnn_model.add(Dense(1, activation='sigmoid'))
    dnn_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc', f1_m, precision_m, recall_m])
    return dnn_model


cv = StratifiedKFold(10, shuffle=True, random_state=42)
clf = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=0)
clf3 = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=3, verbosity=2, random_state=0)
clf4 = LogisticRegression(max_iter=1000, random_state=0)
clf5 = LinearSVC(random_state=0)
clf6 = KerasClassifier(build_fn=dnn_models1, epochs=25, batch_size=1000, verbose=1)

total_scores_1 = []
total_scores_2 = []
total_scores_3 = []
total_scores_4 = []
total_scores_5 = []
total_scores_6 = []
for hitseq_num in tqdm_notebook(range(1,11)):
    온라인_x1, 온라인_y1 = [], []
    for i,j,k in zip(idx, 온라인_x, 온라인_y):
        if i >= hitseq_num:
            온라인_x1.append(j)
            온라인_y1.append(k)
    X_padded1, X_resampled1, Y_resampled1 = make_padding_and_oversample(np.array(온라인_x1), 온라인_y1, length= int(hitseq_num))
    #X_train1, X_test1, y_train1, y_test1 = train_test_split(X_resampled1, Y_resampled1, test_size=0.3, random_state=42)
    
    a_scores = cross_validate(clf, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_1.append(a_scores)
    
    a_scores = cross_validate(clf2, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_2.append(a_scores)
    
    a_scores = cross_validate(clf3, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_3.append(a_scores)
    
    a_scores = cross_validate(clf4, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_4.append(a_scores)
    
    a_scores = cross_validate(clf5, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_5.append(a_scores)
    
    a_scores = cross_validate(clf6, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_6.append(a_scores)

