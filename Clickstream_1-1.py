## 1-1. 현재 세션(1개)의 모든 클릭 로그를 대상으로 LSTM을 사용해서 다음 세션에 구매가 일어날지를 예측  [현준]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

온라인 = pd.read_csv('C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/온라인_전처리_final_32columns.csv', encoding='utf-8')
온라인 = 온라인.sort_values(['clnt_id','sess_id','hit_seq']).reset_index(drop=True)

온라인['unique_id'] = list(map(lambda x,y: str(x)+'_'+str(y), 온라인.clnt_id, 온라인.sess_id))

#다음 세션의 구매를 예측하기 위한 종속변수 생성
구매여부 = 온라인[['clnt_id', 'sess_id', 'buy']].groupby(['clnt_id', 'sess_id']).sum()
구매여부.buy = 구매여부.buy.apply(lambda x:0 if x == 0 else 1)
구매여부 = 구매여부.sort_index()
구매여부 = 구매여부.reset_index()
#g = 구매여부.groupby('clnt_id')

구매여부1 = pd.DataFrame()
for id in tqdm_notebook(구매여부['clnt_id'].unique()):
    temp = 구매여부[구매여부['clnt_id'] == id]
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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical
from keras import backend as K

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
idx1 = 온라인.unique_id.drop_duplicates().index.tolist()
idx2 = idx1[1:] + [len(온라인)]

idx3 = []
for i, j in tqdm_notebook(zip(idx1, idx2), total=len(idx1)):
    temp = 온라인.buy.iloc[i:j]
    try:
        idx3.append(temp[temp == 1].index[0])
    except:
        idx3.append(j)

# (session, sequence, variables) 3d array 변환
온라인_x = []
for i, j in tqdm_notebook(zip(idx1, idx3), total=len(idx1)):
    온라인_x.append(온라인.iloc[i:j, 4:-3].values)
    
#scaler = StandardScaler() 
#scaler = MinMaxScaler(feature_range=(0, 1))
#온라인_x = scaler.fit_transform(np.array(온라인_x))
온라인_x = np.array(온라인_x)

# session 당 구매 여부
온라인_y = []
for i,j in tqdm_notebook(zip(idx1,idx2), total=len(idx1)):
    온라인_y.append([int(온라인.buy.iloc[i:j].sum()>0)])

idx = list(pd.Series(idx3) - pd.Series(idx1))
max(idx), np.percentile(pd.Series(idx),99)

X_padded, X_resampled, Y_resampled = make_padding_and_oversample(온라인_x, 온라인_y, length=max(idx))

'''
X_padded.shape # (294750, 357, 25)
np.savetxt("C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/X_padded.txt", X_padded.reshape((357,-1)), fmt="%s")
X_resampled.shape # (516162, 357, 25)
np.savetxt("C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/X_resampled.txt", X_resampled.reshape((357,-1)), fmt="%s")
len(Y_resampled) # 516162
np.savetxt("C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/Y_resampled.txt", np.array(Y_resampled), fmt="%s")
'''

#X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

import os

import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#export CUDA_VISIBLE_DEVICES=1
KTF.tf.compat.v1.keras.backend.set_session(session)

def models():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64,input_shape = (X_train.shape[1], X_train.shape[2])))
    #model.add(Dense(32, activation= 'relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(16, activation= 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc',f1_m,precision_m, recall_m])
    return model

#model = models(X_train)
#model.summary()

#history = model.fit(X_train, y_train, epochs=25, batch_size=1000, validation_data=(X_test, y_test), verbose=1, shuffle=True)

## Cross-validation
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier


X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
model2 = KerasClassifier(build_fn=models, epochs=25, batch_size=1000, verbose=1)
cv = StratifiedKFold(10, shuffle=True, random_state=42)
acc_scores = cross_validate(model2, X_train, y_train, cv=cv, verbose=1, n_jobs=None, return_train_score=True,
                            scoring=['accuracy', 'f1', 'precision', 'recall'])

if __name__ == "__main__":
    acc_scores = cross_validate(model2, X_train, y_train, cv=cv, verbose=1, n_jobs=-1, return_train_score=True,
                                scoring=['accuracy', 'f1', 'precision', 'recall'])

#y_pred = cross_val_predict(model2, X_test, y_test, cv=10)
#precision, recall, fscore, support = score(y, y_pred, average='micro') 
#print(fscore)


## 1-4. 현재 세션 앞 부분의 1~10개의 클릭 로그를 대상으로 구매 예측을 할 때, LSTM만을 사용해서 구매 예측 [현준]
total_scores_1_10 = []
for hitseq_num in tqdm_notebook(range(1,11)):
    온라인_x1, 온라인_y1 = [], []
    for i,j,k in zip(idx, 온라인_x, 온라인_y):
        if i >= hitseq_num:
            온라인_x1.append(j)
            온라인_y1.append(k)
    X_padded1, X_resampled1, Y_resampled1 = make_padding_and_oversample(np.array(온라인_x1), 온라인_y1, length= int(hitseq_num))
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_resampled1, Y_resampled1, test_size=0.3, random_state=42)
    model2 = KerasClassifier(build_fn=models, epochs=25, batch_size=1000, verbose=1)
    cv = StratifiedKFold(10, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model2, X_train, y_train, cv=cv, verbose=1, n_jobs=None)
    total_scores_1_10.append(acc_scores)








