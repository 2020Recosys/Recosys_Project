import numpy as np 
import pandas as pd
from datetime import datetime
import calendar, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LSTM
from tensorflow.keras.layers import Dropout, Activation, Masking
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop
from tensorflow.keras import backend as K
from sklearn import  metrics


## 클릭스트림 데이터 불러오기
pc_0601 = pd.read_csv("D:/Cheil/preprocessed_data.csv", sep=',')
pc_0601 = pc_0601.drop(['Unnamed: 0'], axis=1)
total_info = pc_0601

## 전체 uid, domain, ownership_1, ownership_2, category_2의 종류와 개수 확인
uid_cv = pc_0601['UID'].value_counts().sort_values(ascending=False)
domain_cv = pc_0601['Domain'].value_counts().sort_values(ascending=False)
o1_cv = pc_0601['ownership_1'].value_counts().sort_values(ascending=False)
o2_cv = pc_0601['ownership_2'].value_counts().sort_values(ascending=False)
c2_cv = pc_0601['category_2'].value_counts().sort_values(ascending=False)

del pc_0601

'''
A3_1(가전제품), A3_2(패션의류/잡화), A3_3(화장품) : 구매했다-1, 구매하지 않았다-2
A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
A4_2_2(구매 시기?) : 초순-1, 중순-2, 하순-3, 구매하지 않음-0  <-- 추후에 구매한 사람과 구매하지 않은 사람을 비교할 필요 있음..
A4_3(구입 경로) : 인터넷쇼핑-4
A4_5(구매경험) : 인터넷-매장-3, 인터넷-인터넷-4, 인터넷&매장-인터넷-6
'''

total_info = pd.read_csv("D:/Cheil/preprocessed_total_info.csv")
total_info = total_info.drop(['Unnamed: 0'], axis=1)

## 설문 데이터 불러오기
pc_survey = pd.read_excel("D:/Cheil/140716_SSK 구매행태 조사 Raw Data_F.xlsx", sep=',')
survey = pc_survey[['UID', 'A3_1', 'A3_2', 'A3_3', 
                    'A4_2_1', 'A4_2_2', 'A4_3', 'A4_5',
                    'A5_2_1', 'A5_2_2', 'A5_3', 'A5_5', 
                    'A6_2_1', 'A6_2_2', 'A6_3', 'A6_5']]
survey = survey.fillna(0)


## A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
survey = pd.concat([survey.where(survey['A4_2_1'] == 3).dropna(), 
                    survey.where(survey['A5_2_1'] == 3).dropna(),
                    survey.where(survey['A6_2_1'] == 3).dropna()]).drop_duplicates()


survey3 = survey[['UID']]





'''
## 선별된 사람들 중 하루에 세션을 가장 많이 열어본 개수를 확인 = 580개
check_max_session = data_하순['Time']
check_max_session = pd.to_datetime(check_max_session).dt.day

check_max_session = data_하순.groupby(['UID'])['session_id'].unique()
check_max_session = pd.DataFrame(check_max_session)
for i in range(len(check_max_session)) :
    check_max_session.iloc[i,0] = len(check_max_session.iloc[i,0])
check_max_session = check_max_session['session_id'].sort_values(ascending=False)
print(max(check_max_session))
check_max_session = max(check_max_session)

# data_초순 : 426개
# data_중순 : 486개
# data_하순 : 580개
'''




###################################
## 컬럼명이 저장된 파일을 불러오기
x_columns = pd.read_csv("D:/Cheil/preprocessed_x_columns3.csv")
x_columns = x_columns.drop(columns=['Unnamed: 0'], axis=1)
x_columns = pd.DataFrame(pd.concat([x_columns.iloc[:25, 0], x_columns.iloc[26:, 0]]))
x_columns = pd.DataFrame([i[0] for i in x_columns.values if i[0] in ['unique_site', 'total_site', 'keyword1', 
                                                                     'mobile', 'pc', 'time_length', 'time_diff', 'holiday_diff', 
                                                                     't_day', 't_buy', 't_hour', 't_minute']]).rename(columns={0 : 'x_columns'})
x_columns.iloc[6, 0] = 'category_hour'
x_columns.iloc[7, 0] = 'holiday'


## 하루 최대 세션 길이 설정 (Padding)
check_max_session = 580


## 세션이 하나도 없는 유저들이 있는 경우, 해당 list에 추가시킴
check_zero_session = []


## X, Y 데이터 생성
X_초순_가전제품 = X_중순_가전제품 = X_하순_가전제품 = X_초순_패션 = X_중순_패션 = X_하순_패션 = X_초순_화장품 = X_중순_화장품 = X_하순_화장품 = pd.DataFrame(np.zeros((0, 12)), columns=list(x_columns.iloc[:,0]))
Y_초순_가전제품 = Y_중순_가전제품 = Y_하순_가전제품 = Y_초순_패션 = Y_중순_패션 = Y_하순_패션 = Y_초순_화장품 = Y_중순_화장품 = Y_하순_화장품 = []


'''
A3_1(가전제품), A3_2(패션의류/잡화), A3_3(화장품) : 구매했다-1, 구매하지 않았다-2
A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
A4_2_2(구매 시기?) : 초순-1, 중순-2, 하순-3, 구매하지 않음-0  <-- 추후에 구매한 사람과 구매하지 않은 사람을 비교할 필요 있음..
A4_3(구입 경로) : 인터넷쇼핑-4
A4_5(구매경험) : 인터넷-매장-3, 인터넷-인터넷-4, 인터넷&매장-인터넷-6
'''
data_초순 = total_info[(pd.to_datetime(total_info['Time'])<= '2014-06-10') & (pd.to_datetime(total_info['Time']) > '2014-05-31')]
data_중순 = total_info[(pd.to_datetime(total_info['Time'])<= '2014-06-20') & (pd.to_datetime(total_info['Time']) > '2014-06-10')]
data_하순 = total_info[(pd.to_datetime(total_info['Time'])<= '2014-06-30') & (pd.to_datetime(total_info['Time']) > '2014-06-20')]


survey4 = survey.where(survey['A4_2_1'] == 3).dropna()
survey_초순_가전제품 = survey4.where(survey4['A4_2_2'] == 1).dropna()
survey_중순_가전제품 = survey4.where(survey4['A4_2_2'] == 2).dropna()
survey_하순_가전제품 = survey4.where(survey4['A4_2_2'] == 3).dropna()

survey5 = survey.where(survey['A5_2_1'] == 3).dropna()
survey_초순_패션 = survey5.where(survey5['A5_2_2'] == 1).dropna()
survey_중순_패션 = survey5.where(survey5['A5_2_2'] == 2).dropna()
survey_하순_패션 = survey5.where(survey5['A5_2_2'] == 3).dropna()

survey6 = survey.where(survey['A6_2_1'] == 3).dropna()
survey_초순_화장품 = survey6.where(survey6['A6_2_2'] == 1).dropna()
survey_중순_화장품 = survey6.where(survey6['A6_2_2'] == 2).dropna()
survey_하순_화장품 = survey6.where(survey6['A6_2_2'] == 3).dropna()    


def check_time(t_time) :
    t_time_n = []
    for i in t_time :
        if i < 6 :
            t_time_n.append(1) # 1 : Midnight
        elif i < 12 and i >= 6 :
            t_time_n.append(2) # 2 : Morning
        elif i < 18 and i >= 12 :
            t_time_n.append(3) # 3 : Afternoon
        else :
            t_time_n.append(4) # 4 : Evening
    return pd.Series(t_time_n, index=t_time.index)

def check_day(t_time) :
    t_time_n = []
    for i in t_time :
        if i in [1, 4, 6, 7, 8, 14, 15, 21, 22, 28, 29] :
            t_time_n.append(1) # 공휴일 (토, 일요일 및 법정 공휴일[ex: 현충일, 지방선거일])
        else :
            t_time_n.append(0) # 평일
    return pd.Series(t_time_n)

def check_buy(t_buy, t_day) :
    for i, i_value in enumerate(t_buy['t_day']) :
        if i_value in t_day :
            t_buy.iloc[i, 1] = 1 # 구매함
        else :
            t_buy.iloc[i, 1] = 0 # 구매하지 않음
    return t_buy

## 설문 대상자 한 명의 클릭스트림 데이터를 기반으로 세션별 변수 값들을 추출 후, full_X 변수에 concat하고, full_Y에 append함
# 설문 대상자 수 만큼 반복
def data_for_LSTM(survey_product, total_info2, full_X, full_Y, check_zero_session) :
    '''
    survey_product = survey_하순_가전제품.copy().reset_index().drop(['index'], axis=1)
    total_info2 = data_하순.copy()
    full_X = X_하순_가전제품.copy()
    full_Y = Y_하순_가전제품.copy()
    uid = survey_product.iloc[34,0]
    uid_index = 34
    '''
    for uid_index, uid in enumerate(survey_product.iloc[:,0]) :
        print("# %d번째" % uid_index)
        
        ## 해당 유저의 클릭스트림을 전체 데이터셋으로부터 추출
        uid_02 = total_info2[total_info2['UID'] == uid]


        ## 세션이 하나도 없는 유저들이 있는 경우, check_zero_session이라는 list에 추가시키고 아래의 코드는 전부 Skip함
        if len(uid_02) == 0 :
            check_zero_session.append((uid_index, uid))
            print("%d번째에 Zero session 존재함" % uid_index)
            continue
    
    
        #1. 세션별 unique 사이트 개수 (Domain 기준)
        e_freq = uid_02.groupby(['session_id'])['Domain'].value_counts()
        e_freq = pd.DataFrame(e_freq)
        e_freq = e_freq.rename(columns={'Domain' : 'Domain_n'})
        e_freq = e_freq.reset_index() # each frequency
        u_freq = e_freq.groupby('session_id').count() # unique frequency
        u_freq = u_freq.drop(['Domain_n'], axis=1)
        u_freq = u_freq.rename(columns={'Domain' : 'unique_site'})
    
    
        #2. 세션별 total 사이트 개수
        t_info = uid_02.groupby('session_id').count() # total information
        t_site = t_info[['Domain']]
        t_site = t_site.rename(columns={'Domain' : 'total_site'})
        
        
        #3. 세션별 total 검색 횟수 (keyword_p랑 keyword_t 각각)
        t1 = t_info[['keyword_p']].rename(columns={'keyword_p' : 'keyword1'})
            
        
        #8. 세션 및 pc/mobile별 total 사이트 개수
        t_pm = uid_02.groupby(['session_id'])['PC'].value_counts()
        t_pm = pd.DataFrame(t_pm)
        t_pm = t_pm.rename(columns={'PC' : 'PC_n'})
        t_pm = t_pm.reset_index()
        t_pm = pd.pivot_table(t_pm, index='session_id', columns='PC', values='PC_n').fillna(0)
        if t_pm.columns[0] == 1 : # Mobile열이 없는 경우
            t_pm['mobile'] = 0
            t_pm = t_pm.rename(columns={1 : 'pc'})
            t_pm = t_pm[['mobile', 'pc']]
        elif len(t_pm.columns) == 1 : # PC열이 없는 경우
            t_pm['pc'] = 0
            t_pm = t_pm.rename(columns={0 : 'mobile'})
        else : 
            t_pm = t_pm.rename(columns={0 : 'mobile', 1 : 'pc'})
        
        
        #9. 세션의 길이 (초 단위, 마지막 사이트 접속 시간 - 처음 사이트 시작 시간)
        t_le_f = uid_02.groupby(['session_id'])['Time'].min()
        t_le_l = uid_02.groupby(['session_id'])['Time'].max()
        t_le_f = pd.to_datetime(t_le_f)
        t_le_l = pd.to_datetime(t_le_l)
        t_le = pd.DataFrame(t_le_l - t_le_f)
        t_le = t_le.rename(columns={'Time' : 'time_length'})
        t_le = t_le['time_length'].dt.total_seconds()
        
        
        #10. 세션 시작 및 종료 시간대 (처음 사이트 시작시간 기준, 오전6-11/오후12-17/저녁18-23/심야0-5)
        t_time_f = t_le_f.dt.hour
        t_time_f = check_time(t_time_f)
        t_time = pd.DataFrame(t_time_f).rename(columns={0 : 'category_hour'})
        
        
        #11. 세션 시작 및 종료 공휴일 여부 (처음 사이트 시작시간이나 마지막 사이트 시작시간이 공휴일에 포함되는지 여부)
        t_holiday_f = t_le_f.dt.day
        t_holiday_l = t_le_l.dt.day
        t_holiday_index = t_holiday_f.index
        t_holiday_f = check_day(t_holiday_f)
        t_holiday_l = check_day(t_holiday_l)
        t_holiday = pd.Series((1 if t_holiday_f[x] == 1 or t_holiday_l[x] == 1 else 0 for x in range(len(t_holiday_f)))) # 공휴일: 1 / 공휴일 아님: 0
        t_holiday = pd.DataFrame(t_holiday, columns=['holiday']).set_index(t_holiday_index)
        del t_holiday_f, t_holiday_l
        
        
        #12. 세션 시작일 (날짜)
        #13. 세션 구매여부 (날짜)
        # A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
        t_day = pd.DataFrame(t_le_f.dt.day)
        t_day = t_day.rename(columns={'Time' : 't_day'})
        
        if t_day.iloc[0, 0] < 11 :
            max_value = 10
            check_value = 0
        elif t_day.iloc[0, 0] < 21 :
            max_value = 20
            check_value = 1
        else :
            max_value = 30
            check_value = 2
        
        uid_02_period = survey_product[survey_product['UID'] == uid]
        uid_month = uid_02_period[['A4_2_1', 'A5_2_1', 'A6_2_1']]
        uid_02_period = uid_02_period[['A4_2_2', 'A5_2_2', 'A6_2_2']]
        uid_02_period = [uid_02_period.iloc[0, check_value] if uid_month.iloc[0, check_value] == 3 else 0]
        
        t_buy = t_day.copy()
        t_buy['t_buy'] = 0
        t_day2 = []
        if 1 in uid_02_period :
            t_day2.append(5)
        if 2 in uid_02_period :
            t_day2.append(15)
        if 3 in uid_02_period :
            t_day2.append(25)
        t_buy = check_buy(t_buy, t_day2)
    
        '''
        #14. 세션 사이트 개수 비율 (unique 사이트 / total 사이트)
        site_ratio = u_freq['unique_site'] / t_site['total_site']
        site_ratio = pd.DataFrame(site_ratio).rename(columns={0 : 'site_ratio'})
        '''
    
        
        ## 독립변수들을 하나의 데이터프레임으로 합치기
        t_le_f2 = pd.DataFrame(t_le_f.dt.hour)
        t_le_f2 = t_le_f2.rename(columns={'Time' : 't_hour'})
        t_le_f3 = pd.DataFrame(t_le_f.dt.minute)
        t_le_f3 = t_le_f3.rename(columns={'Time' : 't_minute'})
        t_c = pd.concat([u_freq, t_site, t1, t_pm, t_le, t_time, t_holiday, t_buy, t_le_f2, t_le_f3], axis=1)
        
        t_c = t_c.sort_values(by=['t_day', 't_hour', 't_minute'], ascending=True)
        t_c = t_c.reset_index().drop(columns=['session_id'])
        
        
        ## Padding 처리
        max_session = 580        
        for j in range(max_session - len(t_c)) :
            t_c = pd.concat([t_c, pd.DataFrame(np.full((1,12), np.nan), columns=t_c.columns)])
            t_c.iloc[len(t_c)-1, 8] = max_value  # day
            t_c.iloc[len(t_c)-1, 10] = 24 # hour
            t_c.iloc[len(t_c)-1, 11] = 60 # minute
            
                
        X = t_c.sort_values(by=['t_day', 't_hour', 't_minute'], ascending=True)        
        
        ## full_X와 full_Y에 해당 설문 대상자의 데이터를 저장
        full_X = pd.concat([full_X, X])
        if t_day2 != [] :
            full_Y.append(1.0)
        else :
            full_Y.append(0.0)
    return check_zero_session, full_X.drop(columns=['t_day', 't_minute', 't_hour', 't_buy']), pd.DataFrame(full_Y, columns=['full_Y'])
            

## 초순
survey_초순_가전제품 = survey_초순_가전제품.reset_index().drop(['index'], axis=1)
check_zero_session, X_초순_가전제품, Y_초순_가전제품 = data_for_LSTM(survey_초순_가전제품, data_초순, X_초순_가전제품, Y_초순_가전제품, check_zero_session)
if check_zero_session != [] :
    survey_초순_가전제품 = survey_초순_가전제품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_초순_패션 = survey_초순_패션.reset_index().drop(['index'], axis=1)
check_zero_session, X_초순_패션, Y_초순_패션 = data_for_LSTM(survey_초순_패션, data_초순, X_초순_패션, Y_초순_패션, check_zero_session)
if check_zero_session != [] :
    survey_초순_패션 = survey_초순_패션.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_초순_화장품 = survey_초순_화장품.reset_index().drop(['index'], axis=1)
check_zero_session, X_초순_화장품, Y_초순_화장품 = data_for_LSTM(survey_초순_화장품, data_초순, X_초순_화장품, Y_초순_화장품, check_zero_session)
if check_zero_session != [] :
    survey_초순_화장품 = survey_초순_화장품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []


## 중순
survey_중순_가전제품 = survey_중순_가전제품.reset_index().drop(['index'], axis=1)
check_zero_session, X_중순_가전제품, Y_중순_가전제품 = data_for_LSTM(survey_중순_가전제품, data_중순, X_중순_가전제품, Y_중순_가전제품, check_zero_session)
if check_zero_session != [] :
    survey_중순_가전제품 = survey_중순_가전제품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_중순_패션 = survey_중순_패션.reset_index().drop(['index'], axis=1)
check_zero_session, X_중순_패션, Y_중순_패션 = data_for_LSTM(survey_중순_패션, data_중순, X_중순_패션, Y_중순_패션, check_zero_session)
if check_zero_session != [] :
    survey_중순_패션 = survey_중순_패션.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_중순_화장품 = survey_중순_화장품.reset_index().drop(['index'], axis=1)
check_zero_session, X_중순_화장품, Y_중순_화장품 = data_for_LSTM(survey_중순_화장품, data_중순, X_중순_화장품, Y_중순_화장품, check_zero_session)
if check_zero_session != [] :
    survey_중순_화장품 = survey_중순_화장품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []
    

## 하순
survey_하순_가전제품 = survey_하순_가전제품.reset_index().drop(['index'], axis=1)
check_zero_session, X_하순_가전제품, Y_하순_가전제품 = data_for_LSTM(survey_하순_가전제품, data_하순, X_하순_가전제품, Y_하순_가전제품, check_zero_session)
if check_zero_session != [] :
    survey_하순_가전제품 = survey_하순_가전제품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_하순_패션 = survey_하순_패션.reset_index().drop(['index'], axis=1)
check_zero_session, X_하순_패션, Y_하순_패션 = data_for_LSTM(survey_하순_패션, data_하순, X_하순_패션, Y_하순_패션, check_zero_session)
if check_zero_session != [] :
    survey_하순_패션 = survey_하순_패션.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

survey_하순_화장품 = survey_하순_화장품.reset_index().drop(['index'], axis=1)
check_zero_session, X_하순_화장품, Y_하순_화장품 = data_for_LSTM(survey_하순_화장품, data_하순, X_하순_화장품, Y_하순_화장품, check_zero_session)
if check_zero_session != [] :
    survey_하순_화장품 = survey_하순_화장품.drop([c_z[0] for c_z in check_zero_session]).reset_index().drop(['index'], axis=1)
    print("session이 하나도 없는 %d명의 유저가 있음(+삭제시킴)" % len(check_zero_session))
    check_zero_session = []

    
## 매 Epoch마다 recall값을 확인하기 위함
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


## Main Model
def Main_Model(full_X2, full_Y2, survey_product, max_session = 580, a_batch_size = 10) :    
    '''
    full_X2 = X_하순_가전제품.copy() 
    full_Y2 = Y_하순_가전제품.copy() 
    survey_product = survey_하순_가전제품.copy()
    max_session = 580
    a_batch_size = 10
    '''
    
    ## Data Scaling : 0과 1사이의 값으로 만듦
    #scaler = StandardScaler() 
    scaler = MinMaxScaler(feature_range=(0, 1))
    full_X2 = scaler.fit_transform(full_X2)
    
    
    ## 결측값을 Masking하기 위해 -9999로 수정
    full_X2 = np.nan_to_num(full_X2, nan=-9999)
    
    
    ## Train size : 70%, Validation size : 20%, Test size : 10%
    train_ratio = 0.70
    validation_ratio = 0.20
    
    train_size = int(len(survey_product) * train_ratio) * max_session
    validation_size = int(len(survey_product) * validation_ratio) * max_session + train_size
        
    train_Y_size = int(len(survey_product) * train_ratio)
    validation_Y_size = int(len(survey_product) * validation_ratio) + train_Y_size
    
    train_X = np.array(full_X2[:train_size, :])
    train_Y = np.array(full_Y2.iloc[:train_Y_size, :])
    
    validation_X = np.array(full_X2[train_size:validation_size, :])
    validation_Y = np.array(full_Y2.iloc[train_Y_size:validation_Y_size, :])
    
    test_X = np.array(full_X2[validation_size:, :])
    test_Y = np.array(full_Y2.iloc[validation_Y_size:, :])
    
    ## Class 불균형 해결을 위해 Class별 가중치 적용
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_Y),
                                                     [train_Y[x][0] for x in range(len(train_Y))])
    class_weights = dict(enumerate(class_weights))
    
    
    ## RNN의 입력값 형태를 계산
    a_samples1 = int(len(survey_product) * train_ratio)
    a_samples2 = int(len(survey_product) * validation_ratio)
    a_samples3 = int(len(survey_product) - a_samples1 - a_samples2)
    a_timesteps = int(max_session)
    a_features = int(train_X.shape[1])
    
    
    ## Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((a_samples1, a_timesteps, a_features))
    validation_X = validation_X.reshape((a_samples2, a_timesteps, a_features))
    test_X = test_X.reshape((a_samples3, a_timesteps, a_features))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    
    
    ## Design network   
    model = Sequential()
    model.add(Masking(mask_value=-9999, input_shape=(a_timesteps, a_features)))
    #model.add(LSTM(64, input_shape=(a_timesteps, a_features)))
    #model.add(Dense(32, activation= 'relu'))
    #model.add(Dense(16, activation= 'relu'))
    model.add(LSTM(64, input_shape=(a_timesteps, a_features), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr= 0.003, rho = 0.9), metrics=['accuracy', recall_m])
    model.compile(loss='binary_crossentropy', optimizer=Adamax(lr= 0.0005), metrics=['accuracy', recall_m])
    
    ## Fit network
    history = model.fit(train_X, train_Y, epochs=20, class_weight=class_weights, batch_size=1, validation_data=(validation_X, validation_Y), shuffle=False)
    
    return history, test_X, test_Y, a_features, model


history, test_X, test_Y, a_features, model = Main_Model(X_하순_가전제품, Y_하순_가전제품, survey_하순_가전제품)

## Sensitivity Analysis
test_X_c = test_X.drop(columns=['t_day', 't_buy']).columns
importance_df = pd.DataFrame(np.zeros((a_features, 2)), columns=['varible_name', 'perturbation_effect'])
def var_importance(model):
    x = test_X # Get a sample of data
    orig_out = model.predict(x)
    for i in range(a_features):  # iterate over the three features
        new_x = x.copy()
        perturbation = np.random.normal(0.0, 0.2, size=new_x.shape[:2])
        new_x[:, :, i] = new_x[:, :, i] + perturbation
        perturbed_out = model.predict(new_x)
        effect = ((orig_out - perturbed_out) ** 2).mean() ** 0.5 #RMSE
        importance_df.iloc[i,0] = test_X_c[i]
        importance_df.iloc[i,1] = effect
        print(f'Variable {i+1}, perturbation effect: {effect:.4f}')
var_importance(model)
importance_df = importance_df.sort_values(by='perturbation_effect', ascending=False)
print(importance_df)
    

## Plot 예시
mpl.rc('font', family='New Gulim')
mpl.rc('axes', unicode_minus=False)
plt.subplots(figsize=(18,5))
plt.bar(importance_df.iloc[:,0], importance_df.iloc[:,1])
plt.title('Sensitivity Analysis')
plt.xlabel('IV')
plt.ylabel('Perturbation Effect')
plt.xticks(importance_df.iloc[:,0], rotation=90)
plt.show()

plt.plot(history.history['accuracy'], 'b', label='acc_train')
plt.plot(history.history['val_accuracy'], 'r', label='acc_val')
plt.plot(history.history['recall_m'], 'b--', label='rec_train')
plt.plot(history.history['val_recall_m'], 'r--', label='rec_val')
plt.title('Accuracy & Recall')
plt.xlabel('Epochs')
plt.ylabel('Accuracy & Recall')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], 'b', label='acc_train')
plt.plot(history.history['val_accuracy'], 'r', label='acc_val')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['recall_m'], 'b--', label='rec_train')
plt.plot(history.history['val_recall_m'], 'r--', label='rec_val')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
sum(train_Y) / len(train_Y)
sum(test_Y) / len(test_Y)


## 모델 평가
loss_c1, acc_c1 = model.evaluate(test_X, test_Y)
print('Test loss: ', loss_c1)
print('Test Accuracy: ', acc_c1)


## 모델 예측
pred = model.predict(test_X)
pred = pred.astype('float64')
print(round(len(pred[pred >= 0.5]) / len(pred), 3))
print(round(sum(full_Y) / len(full_Y), 3))


## 모델 결과 확인
print("Test_Accuracy: %.4f " % metrics.accuracy_score(test_Y, pred.round()))
print("Test_Recall: %.4f " % metrics.recall_score(test_Y, pred.round()))
print("Test_Precision: %.4f " % metrics.precision_score(test_Y, pred.round()))
print("Test_F1_score: %.4f " % metrics.f1_score(test_Y, pred.round()))
print("Test_Confusion Matrix: \n", metrics.confusion_matrix(test_Y, pred.round()))
print(metrics.classification_report(test_Y, pred.round()))


## 모델 및 가중치 저장
model_json = model.to_json()
with open("C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/0219_1_model.json", "w") as json_file : 
    json_file.write(model_json)
model.save_weights("C:/Users/sim-server/Desktop/RecommenderSystems/클릭스트림/0219_1_model.h5")


'''
## ------------------ 추후에 데이터를 불러올 때는 아래의 것들만 불러오면 됨 3
from keras.models import model_from_json
json_file = open("D:/Cheil/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("D:/Cheil/model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer=Adamax(lr=0.0005), metrics=['accuracy'],)
loss_c1, loss_c2 = loaded_model.evaluate(test_X, test_Y)
'''