## 1-2. 현재 세션(1개)의 모든 클릭 로그를 대상으로 MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을
# 사용해서 구매 예측  [예린]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

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


#온라인2.to_csv('D:/Cheil/온라인_전처리_flat.csv', encoding='utf-8')

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

def make_padding_and_oversample1(X, Y):
    smote = SMOTE(random_state=0)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)
    return X_resampled, Y_resampled

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인2.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인2)]
idx = list(pd.Series(idx2) - pd.Series(idx1))
max(idx), np.percentile(pd.Series(idx),99)


온라인_x = 온라인2.iloc[:, 1:-3]
온라인_x = np.array(온라인_x)

# session 당 구매 여부
온라인_y = 온라인2.buy

X_resampled, Y_resampled = make_padding_and_oversample1(온라인_x, 온라인_y)

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

cv = StratifiedKFold(10, shuffle=True, random_state=42)

clf = GaussianNB()
acc_scores1 = cross_validate(clf, X_resampled, Y_resampled, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf2 = DecisionTreeClassifier(random_state=0)
acc_scores2 = cross_validate(clf2, X_resampled, Y_resampled, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf3 = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=2, verbosity=2, random_state=0)
acc_scores3 = cross_validate(clf3, X_resampled, Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf4 = LogisticRegression(max_iter=1000, random_state=0)
acc_scores4 = cross_validate(clf4, X_resampled, Y_resampled, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf5 = LinearSVC(random_state=0)
acc_scores5 = cross_validate(clf5, X_resampled, Y_resampled, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])

clf6 = KerasClassifier(build_fn=dnn_models, epochs=25, batch_size=1000, verbose=1)
acc_scores6 = cross_validate(clf6, X_resampled, Y_resampled, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])


acc_scores = [acc_scores1, acc_scores2, acc_scores3, acc_scores4, acc_scores5, acc_scores6]
acc_col = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

acc_result = pd.DataFrame(np.zeros((len(acc_scores), len(acc_col))), columns=acc_col)
for t_s in range(len(acc_scores)) :
    acc_result.iloc[t_s, 0] = np.mean(acc_scores[t_s][0]['test_accuracy'])
    acc_result.iloc[t_s, 1] = np.mean(acc_scores[t_s][0]['test_f1'])
    acc_result.iloc[t_s, 2] = np.mean(acc_scores[t_s][0]['test_precision'])
    acc_result.iloc[t_s, 3] = np.mean(acc_scores[t_s][0]['test_recall'])
acc_result.to_csv('온라인_1-2.csv', encoding='utf-8')






## 1-3. 현재 세션 앞 부분의 1~10개의 클릭 로그를 대상으로 구매 예측을 할 때, MLP, Gaussian Naive Bayes, Decision Tree, XGBoost,
# Logistic Regression, Linear SVM을 사용해서 구매 예측  [예린]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier

온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()


#각 clnt_id별로 shift로 1행씩 올림
구매여부2 = 구매여부.copy()
구매여부2.buy = 구매여부2.buy.shift(-1)
g = 구매여부2.groupby('clnt_id')
#1행씩 올리면 각 clnt_id별로 마지막 값은 다음 id의 값, 따라서 마지막 행 제거
구매여부2.drop(g.tail(1).index, axis=0, inplace = True)
구매여부2['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 구매여부2.clnt_id, 구매여부2.sess_id))


온라인2 = 온라인.copy()
온라인2.drop(['clnt_id', 'sess_id', 'trans_id', 'buy'], axis=1, inplace=True)

온라인_x = np.array(온라인2)

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인2.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인2)]
idx = list(pd.Series(idx2) - pd.Series(idx1))


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


def to_flat(df):
    cc = df.groupby(['unique_id']).cumcount() + 1
    flat_df = df.set_index(['unique_id', cc]).unstack().sort_index(1, level=1)
    flat_df.columns = ['_'.join(map(str,i)) for i in flat_df.columns]
    flat_df.reset_index(inplace=True)
    return flat_df


def make_padding_and_oversample2(X, length=350) :
    # 머신러닝 분류기에 데이터를 집어넣으려면, flat시켜야 됨
    X_flat = to_flat(pd.DataFrame(X, columns=온라인2.columns))
    print("to_flat 완료")
    
    X_flat1 = X_flat.merge(구매여부2, left_on='unique_id', right_on='unique_id', how='left')
    
    # 다음 세션에 구매가 일어났는지를 예측해야 되기 때문에 해당 세션(행)이 각 unique_id의 마지막 세션이면 어쩔 수 없이 제거함
    X_flat1 = X_flat1.dropna()
    X_flat1.sort_values(by=['clnt_id','sess_id'], inplace=True)
    
    # 종속변수(구매여부) 추출
    Y = X_flat1.buy.astype('int').to_list()
    X_flat1 = X_flat1.iloc[:, 1:-3]

    smote = SMOTE(random_state=0)
    X_resampled, Y_resampled = smote.fit_resample(X_flat1, Y)
    print("smote 완료")
    return np.array(X_resampled), Y_resampled
    

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

#hitseq_num = 1
for hitseq_num in tqdm_notebook(range(1,11)):
    온라인_x1 = []
    for idx_index, idx_value in enumerate(idx) :
        # hitseq_num개 이상의 클릭 로그를 가진 세션만 추출
        if idx_value >= hitseq_num :
            # 구매여부 변수에서 unique한 유저 아이디와 세션 아이디 하나를 가지고 옴
            구매여부_idx = 구매여부.iloc[idx_index, :-1]
            구매여부_idx = str(구매여부_idx[0]) + '_' + str(구매여부_idx[1])
            
            # 위에서 가지고 온 유저 아이디와 세션 아이디가 일치하는 데이터만 추출
            온라인_x_partial = 온라인[온라인['unique_id'] == 구매여부_idx].iloc[:, 3:-1]
            
            # hitseq_num개 이상의 클릭 로그를 가진 세션의 클릭 로그 중에서 hitseq_num까지의 클릭 로그만 사용(추출)
            # hitseq_num개 이후의 클릭 로그는 버림
            온라인_x_partial = np.array(온라인_x_partial[온라인_x_partial['hit_seq'] <= hitseq_num])
            for 온라인_x_value in 온라인_x_partial :
                온라인_x1.append(온라인_x_value)
    
    #a1 = 온라인_x1[:10]
    #b1 = idx[:10]
    
    X_resampled1, Y_resampled1 = make_padding_and_oversample2(np.array(온라인_x1), length= int(hitseq_num))

    a_scores = cross_validate(clf, X_resampled1, Y_resampled1, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_1.append(a_scores)

    a_scores = cross_validate(clf2, X_resampled1, Y_resampled1, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_2.append(a_scores)

    a_scores = cross_validate(clf3, X_resampled1, Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_3.append(a_scores)

    a_scores = cross_validate(clf4, X_resampled1, Y_resampled1, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_4.append(a_scores)

    a_scores = cross_validate(clf5, X_resampled1, Y_resampled1, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_5.append(a_scores)

    a_scores = cross_validate(clf6, X_resampled1, Y_resampled1, cv=cv, verbose=3, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    total_scores_6.append(a_scores)

#total_scores값들을 전부 csv파일 형태로 만들어서 저장하기!
total_scores = [total_scores_1, total_scores_2, total_scores_3, total_scores_4, total_scores_5, total_scores_6]
total_col = ['Accuracy', 'F1-Score', 'Precision', 'Recall'] * 10
total_col2 = []
z = 1
for t_c in total_col :
    total_col2.append(t_c + '_' + str(z))
    if t_c == 'Recall' :
        z += 1

total_result = pd.DataFrame(np.zeros((len(total_scores), len(total_col2))), columns=total_col2)
for t_s in range(len(total_scores)) :
    z = 0
    for t_s2 in range(len(total_scores[t_s])) :
        total_result.iloc[t_s, z] = np.mean(total_scores[t_s][t_s2]['test_accuracy'])
        z += 1

        total_result.iloc[t_s, z] = np.mean(total_scores[t_s][t_s2]['test_f1'])
        z += 1

        total_result.iloc[t_s, z] = np.mean(total_scores[t_s][t_s2]['test_precision'])
        z += 1

        total_result.iloc[t_s, z] = np.mean(total_scores[t_s][t_s2]['test_recall'])
        z += 1
total_result.to_csv('온라인_1-3.csv', encoding='utf-8')
