# Predicting Online Purchase Behavior Using Truncated Session Data
## Comparing the within session and inter session

## 참고 사항
### 1. 다음 세션 구매 예측
1. 현재 세션(1개)의 모든 클릭 로그를 대상으로 LSTM을 사용해서 다음 세션에 구매가 일어날지를 예측
2. 현재 세션(1개)의 모든 클릭 로그를 대상으로 MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을 사용해서 구매 예측
3. 현재 세션 앞 부분의 10개의 클릭 로그를 대상으로 구매 예측을 할 때, MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을 사용해서 구매 예측
4. 현재 세션 앞 부분의 2~10개의 클릭 로그를 대상으로 구매 예측을 할 때, LSTM만을 사용해서 구매 예측

### 2. 현재 세션 구매 예측
1. 현재 세션(1개)의 구매 이전 클릭 로그를 대상으로 LSTM을 사용해서 현재 세션에 구매가 일어날지를 예측
  * (구매가 없으면 해당 세션의 전체 클릭로그가 대상임)
2. 현재 세션(1개)의 구매 이전 클릭 로그를 대상으로 MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을 사용해서 현재 세션에 구매가 일어날지를 예측
  * (구매가 없으면 해당 세션의 전체 클릭로그가 대상임)
3. 현재 세션 앞 부분의 10개의 클릭 로그를 대상으로 구매 예측을 할 때, MLP, Gaussian Naive Bayes, Decision Tree, XGBoost, Logistic Regression, Linear SVM을 사용해서 현재 세션에 구매가 일어날지를 구매 예측
4. 현재 세션 앞 부분의 2~10개의 클릭 로그를 대상으로 구매 예측을 할 때, LSTM만을 사용해서 현재 세션에 구매가 일어날지를 예측
