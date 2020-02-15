import numpy as np 
import pandas as pd
from datetime import datetime
import calendar
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LSTM
from tensorflow.keras.layers import Dropout, Activation, Masking
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import backend as K
from sklearn import  metrics

## 클릭스트림 데이터 불러오기
pc_0601 = pd.read_csv("D:/Cheil/preprocessed_data.csv", sep=',')
pc_0601 = pc_0601.drop(['Unnamed: 0'], axis=1)

## 전체 uid, domain, ownership_1, ownership_2, category_2의 종류와 개수 확인
uid_cv = pc_0601['UID'].value_counts().sort_values(ascending=False)
domain_cv = pc_0601['Domain'].value_counts().sort_values(ascending=False)
o1_cv = pc_0601['ownership_1'].value_counts().sort_values(ascending=False)
o2_cv = pc_0601['ownership_2'].value_counts().sort_values(ascending=False)
c2_cv = pc_0601['category_2'].value_counts().sort_values(ascending=False)


'''
A3_1(가전제품), A3_2(패션의류/잡화), A3_3(화장품) : 구매했다-1, 구매하지 않았다-2
A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
A4_2_2(구매 시기?) : 초순-1, 중순-2, 하순-3, 구매하지 않음-0  <-- 추후에 구매한 사람과 구매하지 않은 사람을 비교할 필요 있음..
A4_3(구입 경로) : 인터넷쇼핑-4
A4_5(구매경험) : 인터넷-매장-3, 인터넷-인터넷-4, 인터넷&매장-인터넷-6
'''


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
'''
survey = pd.concat([survey.where(survey['A4_2_1'] == 3).dropna()]).drop_duplicates()
'''

## A4_5(구매경험) : 인터넷-인터넷-4, 인터넷&매장-인터넷-6
survey2 = pd.concat([survey.where(survey['A4_5'] == 4).dropna(),
                     survey.where(survey['A5_5'] == 6).dropna(),
                     survey.where(survey['A5_5'] == 4).dropna(),
                     survey.where(survey['A4_5'] == 6).dropna(),
                     survey.where(survey['A6_5'] == 4).dropna(),
                     survey.where(survey['A6_5'] == 6).dropna(),
                     ]).drop_duplicates()


## A4_3(구입 경로) : 인터넷쇼핑-4
survey3 = pd.concat([survey2.where(survey['A4_3'] == 4).dropna(),
                     survey2.where(survey['A5_3'] == 4).dropna(),
                     survey2.where(survey['A6_3'] == 4).dropna()
                     ]).drop_duplicates()
survey3 = survey3[['UID']]


## 클릭스트림 데이터의 컬럼명 추출
# 'Del' 컬럼명을 포함시키는 이유는 'Unnamed: 0'을 쉽게 제거하기 위함
col = ['Del']
for x in pc_0601.columns :
    col.append(x)


## 우리가 원하는 형태로 설문에 응답한 사람들을 기존의 클릭스트림 데이터에서 추출하기에는 RAM이 부족해서, 기존의 클릭스트림 데이터는 제거
# 그리고 해당 조건에 맞는 사람들을 찾기 위해서 chunk 활용
del pc_0601


## 데이터를 concat 형식으로 이어붙이기 위해, 기준이 될 행 하나를 불러오기
for chunk in pd.read_csv("D:/Cheil/preprocessed_data.csv", sep=',', names=col, skiprows=1, chunksize=1):
    total_info = chunk # 추후에 제거해야됨
    required_info = total_info.copy()
    break

## 1000000(1백만) 행 단위로 데이터를 끊어서 불러온 후(=chunk), 우리가 원하는 형태로 설문에 응답한 사람들만 선별
chunksize = 1000000
i = 0    
for chunk in pd.read_csv("D:/Cheil/preprocessed_data.csv", sep=',', names=col, skiprows=1, chunksize=chunksize):
    print("-- %d번째 --" % i)
    for x, x_value in enumerate(survey3['UID']) :
        if x_value in chunk['UID'].values :
            chunk_info = chunk[chunk['UID'] == x_value]
            total_info = pd.concat([total_info, chunk_info])
    i += 1

    
## Del 컬럼과 기준이 되었던 행(0번 행)을 제거
total_info = total_info.drop([0])
total_info = total_info.drop(['Del'], axis=1)


## 추후에 데이터를 쉽게 불러오기 위해 데이터 저장
total_info.to_csv("D:/Cheil/preprocessed_total_info.csv")
survey3.to_csv("D:/Cheil/preprocessed_survey3.csv")
o1_cv.to_csv("D:/Cheil/preprocessed_o1_cv.csv")


'''
## ------------------ 추후에 데이터를 불러올 때는 아래의 것들만 불러오면 됨 1
pc_survey = pd.read_excel("D:/Cheil/140716_SSK 구매행태 조사 Raw Data_F.xlsx", sep=',')
survey = pc_survey[['UID', 'A3_1', 'A3_2', 'A3_3', 
                    'A4_2_1', 'A4_2_2', 'A4_3', 'A4_5',
                    'A5_2_1', 'A5_2_2', 'A5_3', 'A5_5', 
                    'A6_2_1', 'A6_2_2', 'A6_3', 'A6_5']]
survey = survey.fillna(0)
survey = pd.concat([survey.where(survey['A4_2_1'] == 3).dropna(), 
                    survey.where(survey['A5_2_1'] == 3).dropna(),
                    survey.where(survey['A6_2_1'] == 3).dropna()]).drop_duplicates()
## A4_5(구매경험) : 인터넷-인터넷-4, 인터넷&매장-인터넷-6
survey2 = pd.concat([survey.where(survey['A4_5'] == 4).dropna(),
                     survey.where(survey['A5_5'] == 6).dropna(),
                     survey.where(survey['A5_5'] == 4).dropna(),
                     survey.where(survey['A4_5'] == 6).dropna(),
                     survey.where(survey['A6_5'] == 4).dropna(),
                     survey.where(survey['A6_5'] == 6).dropna(),
                     ]).drop_duplicates()
## A4_3(구입 경로) : 인터넷쇼핑-4
survey3 = pd.concat([survey2.where(survey['A4_3'] == 4).dropna(),
                     survey2.where(survey['A5_3'] == 4).dropna(),
                     survey2.where(survey['A6_3'] == 4).dropna()
                     ]).drop_duplicates()
survey3 = survey3[['UID']]total_info = pd.read_csv("D:/Cheil/preprocessed_total_info.csv")
total_info = total_info.drop(['Unnamed: 0'], axis=1)
o1_cv = pd.read_csv("D:/Cheil/preprocessed_o1_cv.csv", header=None, names=['ownershp_1'])
'''



'''
## 선별된 사람들 중 하루에 세션을 가장 많이 열어본 개수를 확인 = 55개
check_max_session = total_info['Time']
check_max_session = pd.to_datetime(check_max_session).dt.day
total_info['Time2'] = check_max_session

check_max_session = total_info.groupby(['UID', 'Time'])['session_id'].unique()
check_max_session = pd.DataFrame(check_max_session)
for i in range(len(check_max_session)) :
    check_max_session.iloc[i,0] = len(check_max_session.iloc[i,0])
check_max_session = check_max_session['session_id'].sort_values(ascending=False)
max(check_max_session) # 최대 55개
'''



'''
-- Session별 --
1. 세션별 unique 사이트 개수 (Domain 기준)
2. 세션별 total 사이트 개수
3. 세션별 total 검색 횟수 (keyword_p랑 keyword_t 각각)
4. 세션 및 Ownership1 종류별 total 개수 (One-hot encoding형태)
5. 세션 및 Ownership2 종류 unique 개수
6. 세션별 쇼핑 사이트 방문 빈도 (Ownership2 : 종합쇼핑몰, 소셜커머스, 지불/결제)
7. 세션별 가격 비교 사이트 방문 빈도 (Ownership2 : 가격비교, Category_2 : 쇼핑정보, drop_duplicates())
8. 세션 및 pc/mobile별 total 사이트 개수
9. 세션의 길이 (초 단위, 마지막 사이트 접속 시간 - 처음 사이트 시작 시간)
10. 세션 시작 및 종료 시간대 (처음 사이트 시작 시간 및 종료 시간 기준, 오전6-11/오후12-17/저녁18-23/심야24-5)
11. 세션 시작 및 종료 공휴일 여부 (처음 사이트 시작시간이나 마지막 사이트 시작시간이 공휴일에 포함되는지 여부)
12. 세션 시작일 (날짜)
13. 세션 구매여부 (날짜)
14. 세션 사이트 개수 비율 (unique 사이트 / total 사이트)
15. 세션별 쇼핑 사이트 평균 체류 시간 / 세션 내 페이지(url) 별 평균 체류 시간


-- 독립변수 리스트 --
1. unique_site : 세션별 unique 사이트 개수 (Domain 기준)
2. total_site : 세션별 total 사이트 개수
3. keyword1 : 세션별 total 검색 횟수 (keyword_p)
4. keyword2 : 세션별 total 검색 횟수 (keyword_t)
5. Portal : 세션 및 Ownership1 종류별 total 개수 (One-hot encoding형태)
6. Maker : 5번과 동일
7. Retailer : 5번과 동일
8. Information : 5번과 동일
9. Community and Blog : 5번과 동일
10. News and Media : 5번과 동일
11. Brand : 세션 및 5번과 동일
12. SNS : 5번과 동일
13. unique_ownership2 : 세션 및 Ownership2 종류 unique 개수
14. 종합쇼핑몰 : 세션별 쇼핑 사이트 방문 빈도 (Ownership2 : 종합쇼핑몰, 소셜커머스, 지불/결제)
15. 소셜커머스 : 14번과 동일
16. 지불/결제 : 14번과 동일
17. information_site : 세션별 가격 비교 사이트 방문 빈도 (Ownership2 : 가격비교, Category_2 : 쇼핑정보, drop_duplicates())
18. mobile : 세션 및 pc total 사이트 개수
19. pc : 세션 및 mobile total 사이트 개수
20. time_length : 세션의 길이 (초 단위, 마지막 사이트 접속 시간 - 처음 사이트 시작 시간)
21. time_diff : 세션 시작 시간대와 종료 시간대가 바뀌었는지 여부 (처음 사이트 시작 시작 및 종료 시간 기준, 오전6-11/오후12-17/저녁18-23/심야24-5)
22. holiday_diff : 세션 시작 및 종료 공휴일 여부 (처음 사이트 시작시간이나 마지막 사이트 시작시간이 공휴일에 포함되는지 여부)
         즉, 세션 시작 시간이 평일이고 종료 시간이 공휴일인 경우도 공휴일로 취급하고, 반대로 세션 시작 시간이 공휴일이고 종료 시간이 평일이어도 공휴일로 취급
23. t_day : 세션 시작일 (날짜)
24. t_buy : 세션 구매여부 (날짜). 해당 일자에 상품을 구매했는지 여부.
25. site_ratio : 세션 사이트 개수 비율 (unique 사이트 / total 사이트)
26. t_hour : 세션 시작 시간 (hour)
27. t_minute : 세션 시작 시간 (minute)
28. shop_display_time : 세션별 쇼핑 사이트 평균 체류 시간 / 세션 내 페이지(url) 별 평균 체류 시간

-- 종속변수 리스트--
full_Y : 해당 윈도우(9일치)의 다음날(10일째)에 상품을 구매했는지 여부
'''


###################################
## 25개의 컬럼명이 첨부한 파일을 불러오기
x_columns = pd.read_csv("D:/Cheil/preprocessed_x_columns3.csv")
x_columns = x_columns.drop(columns=['Unnamed: 0'], axis=1).rename(columns={'0' : 'x_columns'})
x_columns = pd.DataFrame(pd.concat([x_columns.iloc[:25, 0], x_columns.iloc[26:, 0]]))

## 하루 최대 세션 길이 설정 (Padding)
check_max_session = 55


## X값(독립변수)들을 저장할 변수 생성
full_X = pd.DataFrame(np.zeros((0, 27)), columns=list(x_columns.iloc[:,0]))


## Y값(종속변수 == 다음날 구매 여부 예측)을 저장할 변수 생성
full_Y = []


## 전체 450명 중에서 100명만 분석
#survey3 = survey3.iloc[:100, :]
check_zero_session = []

## 설문 대상자 한 명의 클릭스트림 데이터를 기반으로 세션별 변수 값들을 추출 후, full_X 변수에 concat하고, full_Y에 append함
# 총 100회(=100명) 반복
for uid_index, uid in enumerate(survey3.iloc[:,0]) :
    print("# %d번째" % uid_index)
    #uid = survey3.iloc[99,0]
    #uid_index = 99
    uid_02 = total_info[total_info['UID'] == uid]
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
    t_keyword1 = t_info[['keyword_p']]
    t_keyword2 = t_info[['keyword_t']]
    t1 = pd.DataFrame(np.where(t_keyword1 == 0, np.nan, t_keyword1), columns=['keyword1']).fillna(0)
    t2 = pd.DataFrame(np.where(t_keyword2 == 0, np.nan, t_keyword2), columns=['keyword2']).fillna(0)
    del t_keyword1, t_keyword2
    
    
    #4. 세션 및 Ownership1 종류별 total 개수 (One-hot encoding형태)
    t_o1 = uid_02.groupby(['session_id'])['ownership_1'].value_counts()
    t_o1 = pd.DataFrame(t_o1)
    t_o1 = t_o1.rename(columns={'ownership_1' : 'ownership'})
    t_o1 = t_o1.reset_index()
    t_o1 = pd.pivot_table(t_o1, index='session_id', columns='ownership_1', values='ownership')
    for i in o1_cv.index.to_list() :
        if i not in t_o1.columns :
            t_o1[i] = np.nan
    t_o1 = t_o1[o1_cv.index.to_list()].fillna(0)
    t_o1_df = pd.DataFrame(np.zeros((len(t_info.index), len(t_o1.columns))), columns=o1_cv.index.to_list())
    for i in range(len(t_o1_df)) :
        if i in t_o1.index :
            t_o1_df.iloc[i,:] = t_o1.loc[i,:]
    del t_o1
    
    
    #5. 세션 및 Ownership2 종류 unique 개수
    u_o2 = uid_02.groupby(['session_id'])['ownership_2'].value_counts()
    u_o2 = pd.DataFrame(u_o2)
    u_o2 = u_o2.rename(columns={'ownership_2' : 'ownership2'})
    u_o2 = u_o2.reset_index()
    u_o2 = pd.DataFrame(u_o2['session_id'].value_counts())
    u_o2_df = pd.DataFrame(np.zeros((len(t_info.index), 1)), columns=['unique_ownership2'])
    for i in range(len(u_o2_df)):
        if i in u_o2.index :
            u_o2_df.iloc[i, :] = u_o2.loc[i,"session_id"]
    del u_o2
    
    
    #6. 세션별 쇼핑 사이트 방문 빈도 (Ownership2 : 종합쇼핑몰, 소셜커머스, 지불/결제)
    t_o2 = uid_02.groupby(['session_id'])['ownership_2'].value_counts()
    t_o2 = pd.DataFrame(t_o2)
    t_o2 = t_o2.rename(columns={'ownership_2' : 'ownership'})
    t_o2 = t_o2.reset_index()
    t_o2 = pd.pivot_table(t_o2, index='session_id', columns='ownership_2', values='ownership')
    t_o2_col = ['종합쇼핑몰', '소셜커머스', '지불/결제']
    for i in t_o2_col :
        if i not in t_o2.columns :
            t_o2[i] = np.nan
    t_o2 = t_o2[t_o2_col].fillna(0)
    t_o2_df = pd.DataFrame(np.zeros((len(t_info.index), len(t_o2.columns))), columns=t_o2_col)
    for i in range(len(t_o2_df)) :
        if i in t_o2.index :
            t_o2_df.iloc[i,:] = t_o2.loc[i,:]
    del t_o2, t_o2_col
    
    
    #7. 세션별 가격 비교 사이트 방문 빈도 (Ownership2 : 가격비교, Category_2 : 쇼핑정보, drop_duplicates())
    t_o2_c1 = uid_02[uid_02['ownership_2'] == '가격비교']
    t_o2_c2 = uid_02[uid_02['category_2'] == '쇼핑정보']
    t_o2_c = pd.concat([t_o2_c1, t_o2_c2]).drop_duplicates()
    t_o2_c = pd.DataFrame(t_o2_c['session_id'].value_counts())
    t_o2_c_df = pd.DataFrame(np.zeros((len(t_info.index), 1)), columns=['information_site'])
    for i in range(len(t_o2_c_df)):
        if i in t_o2_c.index :
            t_o2_c_df.iloc[i, :] = t_o2_c.loc[i,"session_id"]
    del t_o2_c
    
    
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
    t_time_l = t_le_l.dt.hour
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
        return pd.Series(t_time_n)
    t_time_f = check_time(t_time_f)
    t_time_l = check_time(t_time_l)
    t_time = t_time_f - t_time_l
    t_time = pd.Series((0 if x == 0 else 1 for x in t_time)) # 차이 있음: 1 / 차이 없음: 0
    t_time = pd.DataFrame(t_time, columns=['time_diff'])
    del t_time_f, t_time_l
    
    
    #11. 세션 시작 및 종료 공휴일 여부 (처음 사이트 시작시간이나 마지막 사이트 시작시간이 공휴일에 포함되는지 여부)
    t_holiday_f = t_le_f.dt.day
    t_holiday_l = t_le_l.dt.day
    def check_day(t_time) :
        t_time_n = []
        for i in t_time :
            if i in [1, 4, 6, 7, 8, 14, 15, 21, 22, 28, 29] :
                t_time_n.append(1) # 공휴일 (토, 일요일 및 법정 공휴일[ex: 현충일, 지방선거일])
            else :
                t_time_n.append(0) # 평일
        return pd.Series(t_time_n)
    t_holiday_f = check_day(t_holiday_f)
    t_holiday_l = check_day(t_holiday_l)
    t_holiday = pd.Series((1 if t_holiday_f[x] == 1 or t_holiday_l[x] == 1 else 0 for x in range(len(t_holiday_f)))) # 공휴일: 1 / 공휴일 아님: 0
    t_holiday = pd.DataFrame(t_holiday, columns=['holiday_diff'])
    del t_holiday_f, t_holiday_l
    
    
    #12. 세션 시작일 (날짜)
    #13. 세션 구매여부 (날짜)
    # A4_2_1(몇 월에 구매?) : 4월-1, 5월-2, 6월-3, 7월-4
    t_day = pd.DataFrame(t_le_f.dt.day)
    t_day = t_day.rename(columns={'Time' : 't_day'})
    uid_02_period = survey[survey['UID'] == uid]
    
    # 
    uid_month = uid_02_period[['A4_2_1', 'A5_2_1', 'A6_2_1']]
    uid_02_period = uid_02_period[['A4_2_2', 'A5_2_2', 'A6_2_2']]
    uid_02_period = [int(uid_02_period.iloc[0, x]) if uid_month.iloc[0, x] == 3 else 0 for x in range(len(uid_02_period.columns))]
    '''
    uid_02_period = uid_02_period[['A4_2_2']]
    uid_02_period = [int(uid_02_period.iloc[0, 0])]
    '''
    t_buy = t_day.copy()
    t_buy['t_buy'] = 0
    t_day = []
    if 1 in uid_02_period :
        t_day.append(3)
        t_day.append(4)
        t_day.append(5)
    if 2 in uid_02_period :
        t_day.append(13)
        t_day.append(13)
        t_day.append(15)
    if 3 in uid_02_period :
        t_day.append(23)
        t_day.append(24)
        t_day.append(25)
    def check_buy(t_buy, t_day) :
        for i, i_value in enumerate(t_buy['t_day']) :
            if i_value in t_day :
                t_buy.iloc[i, 1] = 1 # 구매함
            else :
                t_buy.iloc[i, 1] = 0 # 구매하지 않음
        return t_buy
    t_buy = check_buy(t_buy, t_day)
    # survey.where(survey['A4_2_1'] == 3)

    #14. 세션 사이트 개수 비율 (unique 사이트 / total 사이트)
    site_ratio = u_freq['unique_site'] / t_site['total_site']
    site_ratio = pd.DataFrame(site_ratio).rename(columns={0 : 'site_ratio'})

    '''
    #15. 세션별 쇼핑 사이트 평균 체류 시간 / 세션 내 페이지(url) 별 평균 체류 시간
    #t_total_copy = uid_02.copy()
    t_total_copy = uid_02[['session_id', 'Time', 'category_2', 'ownership_2']]
    for i, i_value in enumerate(t_total_copy['Time']) :
        t_total_copy.iloc[i, 1] = datetime.strptime(i_value, "%Y-%m-%d %H:%M:%S").timestamp()
    t_list1 = pd.DataFrame(t_total_copy.iloc[1:, 1]).set_index(t_total_copy.index[:-1]) # Next Time
    t_list2 = pd.DataFrame(t_total_copy.iloc[:-1, 1]).set_index(t_total_copy.index[:-1]) # Current Time
    t_list1 = t_list1 - t_list2
    t_total_copy['display_time'] = t_list1
    t_total_copy.iloc[-1, 4] = t_total_copy.iloc[-2, 4]
    t_total_mean = pd.DataFrame(t_total_copy.groupby(['session_id'])['display_time'].mean())
    
    t_shopping_mean = t_total_copy.loc[((t_total_copy.ownership_2 == '종합쇼핑몰') | (t_total_copy.ownership_2 == '소셜커머스') | (t_total_copy.ownership_2 == '지불/결제')), ('session_id', 'display_time')]
    t_shopping_mean = pd.DataFrame(t_shopping_mean.groupby(['session_id'])['display_time'].mean())
    t_shopping_list = []
    for i in range(len(u_freq)) :
        if i in t_shopping_mean.index :
            t_shopping_list.append(t_shopping_mean.loc[i, 'display_time'])
        else :
            t_shopping_list.append(0.0)
    t_shopping_list = pd.DataFrame(t_shopping_list).rename(columns={0 : 'shop_display_time'})
    t_shopping_list = pd.DataFrame(t_shopping_list['shop_display_time'] / t_total_mean['display_time']).rename(columns={0 : 'shop_display_time'})
    '''
    
    ## 독립변수들을 하나의 데이터프레임으로 합치기
    t_total = pd.concat([u_freq, t_site, t1, t2, t_o1_df, u_o2_df, t_o2_df, t_o2_c_df, t_pm, t_le, t_time, t_holiday, t_buy, site_ratio], axis=1)
    t_le_f2 = pd.DataFrame(t_le_f.dt.hour)
    t_le_f2 = t_le_f2.rename(columns={'Time' : 't_hour'})
    t_le_f3 = pd.DataFrame(t_le_f.dt.minute)
    t_le_f3 = t_le_f3.rename(columns={'Time' : 't_minute'})
    t_c = pd.concat([u_freq, t_site, t1, t2, t_o1_df, u_o2_df, t_o2_df, t_o2_c_df, t_pm, t_le, t_time, t_holiday, t_buy, site_ratio, t_le_f2, t_le_f3], axis=1)
    
    t_c = t_c.sort_values(by=['t_day', 't_hour', 't_minute'], ascending=True)
    t_c = t_c.reset_index().drop(columns=['index'])
    
  
    
    ## Padding 처리
    max_session = 55
    window_size = 9
    for i in range(1, calendar.monthrange(2014, 6)[1] + 1) :
        try :
            check_session = t_total['t_day'].value_counts().reset_index().sort_values(by='index').set_index('index').loc[i,:][0]
        except :
            check_session = 0
        for j in range(max_session - check_session) :
            t_c = pd.concat([t_c, pd.DataFrame(np.full((1,27), np.nan), columns=t_c.columns)])
            t_c.iloc[len(t_c)-1, 22] = i  # day
            t_c.iloc[len(t_c)-1, 25] = 24 # hour
            t_c.iloc[len(t_c)-1, 26] = 60 # minute
            
            if i in t_day :
                t_c.iloc[len(t_c)-1, 23] = 1
            else : 
                t_c.iloc[len(t_c)-1, 23] = 0
            
    X = t_c.sort_values(by=['t_day', 't_hour', 't_minute'], ascending=True)
    for i in range(len(X)) :
        if X.iloc[i, 0] == np.nan :
            X.iloc[i, 22] = np.nan # day
            X.iloc[i, 25] = np.nan # hour
            X.iloc[i, 26] = np.nan # minute
    
   
    ## full_X와 full_Y에 해당 설문 대상자의 데이터를 저장
    for i in range(0, calendar.monthrange(2014, 6)[1] - window_size) :
        X_session = X.iloc[(i * max_session) : (i + window_size) * max_session, :]
        full_X = pd.concat([full_X, X_session])
        full_Y.append(X.iloc[(i + window_size) * max_session + 1, 23])
    


## 기준이 되었던 행(0번 행)을 제거하고, full_Y를 list에서 데이터 프레임으로 변환
full_X2 = full_X.copy()
#full_X2 = full_X2.iloc[1:,:]
full_Y2 = pd.DataFrame(full_Y, columns=['full_Y'])

#full_X2 = full_X2.iloc[:4656960, :]
#full_Y2 = full_Y.iloc[:9408, :]
full_X2 = full_X2.drop(columns=['t_buy'])

## Data Scaling
#scaler = StandardScaler() 
scaler = MinMaxScaler(feature_range=(0, 1))
full_X2 = scaler.fit_transform(full_X2)


## 결측값을 Masking하기 위해 -9999로 수정
full_X2 = np.nan_to_num(full_X2, nan=-9999)


## Train size : 75%, Test size : 25%
# 추후에 Train, Validation, Test로 나눠서 분석해야됨
train_ratio = 0.75
train_size = int(len(full_X2) * train_ratio)
if train_size % 21 != 0 :
    t_size = train_size % 21
    train_size -= t_size
train_Y_size = int(len(full_Y2) * train_ratio)

train_X = np.array(full_X2[:train_size, :])
train_Y = np.array(full_Y2.iloc[:train_Y_size, :])

test_X = np.array(full_X2[train_size:, :])
test_Y = np.array(full_Y2.iloc[train_Y_size:, :])

## Class 불균형 해결을 위해 Class별 가중치 적용
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_Y),
                                                 [train_Y[x][0] for x in range(len(train_Y))])
class_weights = dict(enumerate(class_weights))

## RNN의 입력값 형태를 계산
#a_samples1 = int((calendar.monthrange(2014, 6)[1] - window_size) * ((len(survey3) - 2) * train_ratio))
#a_samples2 = int((calendar.monthrange(2014, 6)[1] - window_size) * ((len(survey3) - 2) * (1 - train_ratio)))
a_samples1 = int((calendar.monthrange(2014, 6)[1] - window_size) * (len(survey3) * train_ratio))
a_samples2 = int((calendar.monthrange(2014, 6)[1] - window_size) * (len(survey3) * (1 - train_ratio)))
a_timesteps = int(window_size * max_session)
a_features = int(train_X.shape[1])
a_batch_size = 21


## Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((a_samples1, a_timesteps, a_features))
test_X = test_X.reshape((a_samples2, a_timesteps, a_features))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


## Design network
model = Sequential()
model.add(Masking(mask_value=-9999, input_shape=(a_timesteps, a_features)))
model.add(LSTM(64, input_shape=(a_timesteps, a_features), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adamax(lr=0.0005), metrics=['accuracy', recall_m],)


## Fit network
history = model.fit(train_X, train_Y, epochs=10, class_weight=class_weights, batch_size=a_batch_size, validation_data=(test_X, test_Y), shuffle=False)


## Sensitivity Analysis
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
        importance_df.iloc[i,0] = full_X.columns[i]
        importance_df.iloc[i,1] = effect
        print(f'Variable {i+1}, perturbation effect: {effect:.4f}')
var_importance(model)
importance_df = importance_df.sort_values(by='perturbation_effect', ascending=False)
print(importance_df)
    

## Plot 예시
mpl.rc('font', family='New Gulim')
mpl.rc('axes', unicode_minus=False)

plt.bar(importance_df.iloc[:,0], importance_df.iloc[:,1])
plt.title('Sensitivity Analysis')
plt.xlabel('IV')
plt.ylabel('Perturbation Effect')
plt.xticks(importance_df.iloc[:,0], rotation=90)
plt.show()

plt.plot(history.history['accuracy'], 'b', label='acc_train')
plt.plot(history.history['val_accuracy'], 'r', label='acc_test')
plt.plot(history.history['recall_m'], 'b--', label='rec_train')
plt.plot(history.history['val_recall_m'], 'r--', label='rec_test')
plt.title('Accuracy & Recall')
plt.xlabel('Epochs')
plt.ylabel('Accuracy & Recall')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
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
with open("D:/Cheil/model.json", "w") as json_file : 
    json_file.write(model_json)
model.save_weights("D:/Cheil/model.h5")


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