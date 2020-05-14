## 1-1. 현재 세션(1개)의 모든 클릭 로그를 대상으로 LSTM을 사용해서 다음 세션에 구매가 일어날지를 예측  [현준]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()


구매여부1 = pd.DataFrame()
for id in tqdm_notebook(구매여부['clnt_id'].unique()):
    temp = 구매여부[구매여부['clnt_id'] == id].copy()
    temp.buy = temp.buy.shift(-1)
    temp = temp.dropna(axis = 0)
    구매여부1 = pd.concat([구매여부1, temp])

온라인.drop(['buy'], axis =1, inplace= True)
온라인 = pd.merge(온라인, 구매여부1, left_on = ['clnt_id', 'sess_id'], right_on = ['clnt_id', 'sess_id'])

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers import Masking
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, BorderlineSMOTE
from imblearn.over_sampling import *
from imblearn.base import SamplerMixin
from keras.utils import to_categorical
from keras import backend as K
from imblearn.pipeline import Pipeline, make_pipeline

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
    X_padding = sequence.pad_sequences(X, maxlen=length, padding='pre', truncating='post')
    X_padding2 = X_padding.reshape(X.shape[0], length* X_padding.shape[2])
    print("pad_sequences 완료")
    return X_padding, X_padding2, Y_resampled

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인)]


# (session, sequence, variables) 3d array 변환
온라인_x = []
for i, j in tqdm_notebook(zip(idx1, idx2), total=len(idx1)):
    온라인_x.append(온라인.iloc[i:j, 3:-2].values)


온라인_x = np.array(온라인_x)

# session 당 구매 여부
온라인_y = []
for i,j in tqdm_notebook(zip(idx1,idx2), total=len(idx1)):
    온라인_y.append([int(온라인.buy.iloc[i:j].sum()>0)])

idx = list(pd.Series(idx2) - pd.Series(idx1))
max(idx), np.percentile(pd.Series(idx),99)

X_padded, X_resampled, Y_resampled = make_padding_and_oversample(온라인_x, 온라인_y, length=max(idx))


def models():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(X_padded.shape[1], X_padded.shape[2])))
    model.add(LSTM(64,input_shape = (X_padded.shape[1], X_padded.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc', f1_m, precision_m, recall_m])
    return model


## Cross-validation
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier

class SMOTE2(SMOTE):   
    def fit_resample(self, X, y):
        X_shape1 = X.shape[1]
        X_shape2 = X.shape[2]
        X = X.reshape(X.shape[0], X_shape1 * X_shape2)
        smote = SMOTE(random_state=0)
        X, y = smote.fit_resample(X, y)
        X = X.reshape(X.shape[0], X_shape1, X_shape2)
        print("사이즈: %d" % X.shape[0])
        return X, y

model2 = make_pipeline(SMOTE2(random_state=0), KerasClassifier(build_fn=models, epochs=25, batch_size=1000, verbose=1))
cv = StratifiedKFold(10, shuffle=True, random_state=42)
acc_scores = cross_validate(model2, X_resampled.reshape(X_padded.shape[0], X_padded.shape[1], X_padded.shape[2]), Y_resampled, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                            return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
print(acc_scores)

acc_col = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
acc_result = pd.DataFrame(np.zeros((len(acc_scores), len(acc_col))), columns=acc_col)
acc_result.iloc[0, 0] = np.mean(acc_scores['test_accuracy'])
acc_result.iloc[0, 1] = np.mean(acc_scores['test_f1'])
acc_result.iloc[0, 2] = np.mean(acc_scores['test_precision'])
acc_result.iloc[0, 3] = np.mean(acc_scores['test_recall'])


acc_result.to_csv('온라인_1-1.csv', encoding='utf-8')










## 1-4. 현재 세션 앞 부분의 1~10개의 클릭 로그를 대상으로 구매 예측을 할 때, LSTM만을 사용해서 구매 예측 [현준]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#온라인 = pd.read_csv('C:/Users/JKKIM/Desktop/Recommender/온라인_전처리_final_32columns/온라인_전처리_final_32columns.csv', encoding='utf-8-sig')
온라인 = pd.read_csv('./온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()


구매여부1 = pd.DataFrame()
for id in tqdm_notebook(구매여부['clnt_id'].unique()):
    temp = 구매여부[구매여부['clnt_id'] == id].copy()
    temp.buy = temp.buy.shift(-1)
    temp = temp.dropna(axis = 0)
    구매여부1 = pd.concat([구매여부1, temp])

온라인.drop(['buy'], axis =1, inplace= True)
온라인 = pd.merge(온라인, 구매여부1, left_on = ['clnt_id', 'sess_id'], right_on = ['clnt_id', 'sess_id'])

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers import Masking
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, BorderlineSMOTE
from imblearn.over_sampling import *
from imblearn.base import SamplerMixin
from keras.utils import to_categorical
from keras import backend as K
from imblearn.pipeline import Pipeline, make_pipeline

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
    X_padding = sequence.pad_sequences(X, maxlen=length, padding='pre', truncating='post')
    X_padding2 = X_padding.reshape(X.shape[0], length* X_padding.shape[2])
    print("pad_sequences 완료")
    return X_padding, X_padding2, Y

# 각 clnt_id별 session이 바뀌는 지점 index 저장
idx1 = 온라인.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인)]


# (session, sequence, variables) 3d array 변환
온라인_x = []
for i, j in tqdm_notebook(zip(idx1, idx2), total=len(idx1)):
    온라인_x.append(온라인.iloc[i:j, 3:-2].values)


온라인_x = np.array(온라인_x)

# session 당 구매 여부
온라인_y = []
for i,j in tqdm_notebook(zip(idx1,idx2), total=len(idx1)):
    온라인_y.append([int(온라인.buy.iloc[i:j].sum()>0)])

idx = list(pd.Series(idx2) - pd.Series(idx1))
max(idx), np.percentile(pd.Series(idx),99)



def models1():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(X_padded1.shape[1], X_padded1.shape[2])))
    model.add(LSTM(64,input_shape = (X_padded1.shape[1], X_padded1.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc',f1_m,precision_m, recall_m])
    return model

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier

class SMOTE2(SMOTE):   
    def fit_resample(self, X, y):
        X_shape1 = X.shape[1]
        X_shape2 = X.shape[2]
        X = X.reshape(X.shape[0], X_shape1 * X_shape2)
        smote = SMOTE(random_state=0)
        X, y = smote.fit_resample(X, y)
        X = X.reshape(X.shape[0], X_shape1, X_shape2)
        print("사이즈: %d" % X.shape[0])
        return X, y
        

total_scores_1_10 = []
for hitseq_num in tqdm_notebook(range(1,11)):
    hitseq_num = 2
    print("%d번째 시작" % hitseq_num)
    온라인_x1, 온라인_y1 = [], []
    for i,j,k in zip(idx, 온라인_x, 온라인_y):
        if i >= hitseq_num:
            온라인_x1.append(j)
            온라인_y1.append(k)

    print("%d번째 padding 시작" % hitseq_num)
    X_padded1, X_resampled1, Y_resampled1 = make_padding_and_oversample(np.array(온라인_x1), np.array(온라인_y1), length=int(hitseq_num))
    Y_resampled1 = np.ravel(Y_resampled1, order='C')
    
    print("%d번째 train/testp split 시작" % hitseq_num)
    model2 = make_pipeline(SMOTE2(random_state=0), KerasClassifier(build_fn=models1, epochs=25, batch_size=1000, verbose=1))
    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    acc_scores = cross_validate(model2, X_resampled1.reshape(X_padded1.shape[0], X_padded1.shape[1], X_padded1.shape[2]), Y_resampled1, cv=cv, verbose=2, n_jobs=None, return_train_score=True,
                                    return_estimator=True, scoring=['accuracy', 'f1', 'precision', 'recall'])
    
    total_scores_1_10.append(acc_scores)


total_col = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
total_result = pd.DataFrame(np.zeros((10, len(total_col))), columns=total_col)
for t_s in range(len(total_scores_1_10)) :
    total_result.iloc[t_s, 0] = np.mean(total_scores_1_10[t_s]['test_accuracy'])
    total_result.iloc[t_s, 1] = np.mean(total_scores_1_10[t_s]['test_f1'])
    total_result.iloc[t_s, 2] = np.mean(total_scores_1_10[t_s]['test_precision'])
    total_result.iloc[t_s, 3] = np.mean(total_scores_1_10[t_s]['test_recall'])

total_result.to_csv('온라인_1-4.csv', encoding='utf-8')
