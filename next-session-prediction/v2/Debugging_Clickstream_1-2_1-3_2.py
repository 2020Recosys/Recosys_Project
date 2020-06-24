#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[ ]:


import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Masking
from keras.optimizers import RMSprop
import xgboost as xgb
from keras import backend as K


# In[ ]:


X_resampled = pd.read_csv('./X_resampled_1-2.csv')
print("X_resampled loaded!")
Y_resampled = pd.read_csv('./Y_resampled_1-2.csv')
Y_resampled = np.array(Y_resampled)
print("Y_resampled loaded!")

# sample data
# X_resampled = np.random.randn(100, 10)
# Y_resampled = [0] * 70 + [1] * 30
# Y_resampled = np.array(Y_resampled)


# In[ ]:


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

def dnn_models():
    dnn_model = Sequential()
    dnn_model.add(Dense(32, activation='relu', input_shape=(X_resampled.shape[1],)))
    dnn_model.add(Dense(16, activation='relu'))
    dnn_model.add(Dense(1, activation='sigmoid'))
    dnn_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.001, rho = 0.9), metrics=['acc', f1_m, precision_m, recall_m])
    return dnn_model


# In[ ]:


## Cross-validation
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.pipeline import make_pipeline


# In[ ]:


cv = StratifiedKFold(10, shuffle=True, random_state=42).split(X_resampled, Y_resampled)
print("Made cv!")


# In[ ]:


def cv_for_one(algorithm):
    clf = make_pipeline(SMOTE(random_state=0), algorithm)

    # scores basket
    accuracy = []
    f1 = []
    precision = []
    recall = []
    
    idx = 1
    for train_index, test_index in cv:
        print("cv = ", idx)
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = np.take(Y_resampled, train_index, axis=0), np.take(Y_resampled, test_index, axis=0)
        
        # model train, test
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=y_test, y_pred=y_pred))

        # initalize
        X_train, X_test = [], []
        y_train, y_test = [], []
        idx += 1
    
    print("---"*8)
    print("Accuracy:", accuracy)
    print("---"*8)
    print("F1:", f1)
    print("---"*8)
    print("Precision:", precision)
    print("---"*8)
    print("Recall:", recall)
    print("---"*8)
    return accuracy, f1, precision, recall


# In[ ]:


acc_clf1, f1_clf1, precision_clf1, recall_clf1 = cv_for_one(algorithm=GaussianNB())


# In[ ]:


acc_clf2, f1_clf2, precision_clf2, recall_clf2 = cv_for_one(algorithm=DecisionTreeClassifier(random_state=0))


# In[ ]:


# acc_clf3, f1_clf3, precision_clf3, recall_clf3 = cv_for_one(algorithm=xgb.XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=3, verbosity=2, random_state=0))


# In[ ]:


# acc_clf4, f1_clf4, precision_clf4, recall_clf4 = cv_for_one(algorithm=LogisticRegression(max_iter=1000, random_state=0))


# In[ ]:


# acc_clf5, f1_clf5, precision_clf5, recall_clf5 = cv_for_one(algorithm=LinearSVC(random_state=0))


# In[ ]:


# acc_clf6, f1_clf6, precision_clf6, recall_clf6 = cv_for_one(algorithm=KerasClassifier(build_fn=dnn_models, epochs=25, batch_size=1000, verbose=1))

