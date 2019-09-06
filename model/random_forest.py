# Mullin/model/random_forest.py

from module.path_header import *  # 경로 정리해둔 헤더 파일
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import pickle
import time

TRAIN_NAME = '0906-09-train-v2.csv'
LABEL_NAME = 'train_label.csv'
TEST1_NAME = '0906-09-test1-v2.csv'
TEST2_NAME = '0906-09-test2-v2.csv'
MODEL1_NAME = 'random_forest_model1.pkl'
MODEL2_NAME = 'random_forest_model2.pkl'

PREDICT1_NAME = 'test1_predict.csv'
PREDICT2_NAME = 'test2_predict.csv'

TRAIN_PATH = os.path.join(PREPROCESS_DIR, TRAIN_NAME) 
LABEL_PATH = os.path.join(PREPROCESS_DIR, LABEL_NAME)
TEST1_PATH = os.path.join(PREPROCESS_DIR, TEST1_NAME)
TEST2_PATH = os.path.join(PREPROCESS_DIR, TEST2_NAME)
MODEL1_PATH = os.path.join(MODEL_DIR, MODEL1_NAME)  # survival_time prediction model
MODEL2_PATH = os.path.join(MODEL_DIR, MODEL2_NAME)  # amount_spent prediction model

PREDICT1_PATH = os.path.join(PREDICT_DIR, PREDICT1_NAME)
PREDICT2_PATH = os.path.join(PREDICT_DIR, PREDICT2_NAME)

## main function
# survival_time, amount_spent 에 대한 모델 각각 만들고 model/ 에 저장한다.
# size=40000 (전체 train dataset) 으로 하면 시간 오래걸린다.
def create_model_rf(train_X, train_y, size=1000):    
    # train_test_split
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=0.8)     
    
    amo_cols = ['payment_logged_in', 'tot_spent', 'max_spent', 'amount_spent',
       'mean_spent', 'median_spent', 'a_fish', 'min_spent', 'fishing_prop',
       'a_boss_monster', 'get_item_amount', 'tot_c_rank_per_p', 'p_c_cnt',
       'tot_start_lv', 'tot_end_lv', 'p_same_pledge_cnt',
       'p_random_attacker_cnt', 'count_get', 'p_temp_cnt', 'c_num_opponent',
       'p_etc_cnt', 'p_c_char_cnt', 'c_pledge_cnt', 'pledge_num_people',
       'c_temp_cnt', 'a_quest_exp', 'trade_type_1', 'get_item_price',
       'c_etc_cnt', 'a_npc_kill', 'avg_play_rate_rank_per_p']

    sur_cols = ['activity_logged_in', 'combat_logged_in', 'a_playtime',
       'trade_logged_in', 'sell_item_price', 'count_sell', 'sell_item_amount',
       'combat_count', 'trade_type_0', 'get_item_price', 'trade_time_bin_3',
       'trade_time_bin_2', 'tot_trade_amount', 'total_trade_count',
       'a_private_shop', 'trade_time_bin_0', 'trade_time_bin_1', 'count_get',
       'a_fish', 'trade_type_1', 'a_boss_monster', 'fishing_prop',
       'get_item_amount', 'tot_start_lv', 'pledge_logged_in', 'c_temp_cnt',
       'tot_end_lv', 'a_npc_kill', 'avg_play_rate_rank_per_p',
       'p_play_char_cnt', 'a_quest_exp', 'changed_lv', 'qexp_per_playtime',
       'a_solo_exp', 'sexp_per_playtime', 'tot_exp']

    print('create survival time model')
    survival_time_model(size, train_X, val_X, train_y, val_y)
    print('create amount spent model')
    amount_spent_model(size, train_X, val_X, train_y, val_y)
    

# random_forest 에 맞게 train input 형태 조정
# 1~28 day 무시하고 acc_id 에 대한 값으로 squeeze
def preprocess_X(train_X):
    ## 합할 feature, 평균낼 feature 나누기
    # to mean features
    mean_features = ['tot_c_rank_per_p']
    # to sum features
    sum_features = train_X.columns.tolist()[2:]
    for feat in mean_features:
        sum_features.remove(feat)
    # acc_id 에 대해 mean_features, sum_features 컬럼 평균/합 한 value들 concat 하기
    # 의미: 1~28day 무시하고 feature들을 acc_id 에 대한 값으로 squeeze 하기
    mean_pivot = train_X.pivot_table(index='acc_id', values=mean_features, aggfunc='mean')
    sum_pivot = train_X.pivot_table(index='acc_id', values=sum_features, aggfunc='sum')
    train_X = pd.concat((mean_pivot, sum_pivot), axis=1)        

    # reset_index + acc_id 컬럼 지우기 (acc_id 인덱스, 따로 순서대로 저장해서 필요x)
    train_X = train_X.reset_index(drop=True)

    return train_X

def preprocess_y_survival(train_y):
    train_y = train_y.iloc[:,1]  # survival_time column 추출
    # random_forest label 에 넣기 위해 스칼라 값으로 reshape
    train_y = train_y.values.reshape(-1,) # reshape (40000, )
    return train_y

def preprocess_y_spent(train_y):
    train_y = train_y.iloc[:,-1]  # amount_spent column 추출
    # random_forest label 에 넣기 위해 스칼라 값으로 reshape
    train_y = train_y.values.reshape(-1,) # reshape (40000, )
    return train_y



def survival_time_model(size, train_X, val_X, train_y, val_y):
    train_y = preprocess_y_survival(train_y)
    val_y = preprocess_y_survival(val_y)

    rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rfc.fit(train_X[:size], train_y[:size])

    score = rfc.score(val_X, val_y)
    predict = rfc.predict(val_X)
    
    print(f'validation dataset 에 대한 score: {score:.4f}')
    print(f'validation dataset 의 분류된 label 수: {len(np.unique(predict))}')
    print()
    
    # 모델 저장 (model/*_model1.pkl)
    with open(MODEL1_PATH, 'wb') as fp:
        pickle.dump(rfc, fp)
        
def amount_spent_model(size, train_X, val_X, train_y, val_y):
    train_y = preprocess_y_spent(train_y)
    val_y = preprocess_y_spent(val_y)

    rfr = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rfr.fit(train_X[:size], train_y[:size])

    score = rfr.score(val_X, val_y)
    predict = rfr.predict(val_X)
    mse = mean_squared_error(val_y, predict)
    
    print(f'validation dataset 에 대한 mse score: {mse:.4f}')
    print()
    
    # 모델 저장 (model/*_model2.pkl)
    with open(MODEL2_PATH, 'wb') as fp:
        pickle.dump(rfr, fp)

# 저장된 모델 불러와서 test dataset 에 대해 예측
def test_model_rf(train, test1, test2):  
    train_y1 = preprocess_y_survival(pd.read_csv(LABEL_PATH))
    train_y2 = preprocess_y_spent(pd.read_csv(LABEL_PATH))

    test1_ids = np.unique(pd.read_csv(TEST1_PATH).acc_id)
    test2_ids = np.unique(pd.read_csv(TEST2_PATH).acc_id)



    with open(MODEL1_PATH, 'rb') as fp:
        model = pickle.load(fp)
        score = model.score(train, train_y1)
        s_predict1 = model.predict(test1)
        s_predict2 = model.predict(test2)
        
        print('survival time')
        print(f'전체 train dataset 에 대한 score: {score:.4f}')
        print(f'분류된 test1 dataset 의 label 수: {len(np.unique(s_predict1))}')
        print(f'분류된 test2 dataset 의 label 수: {len(np.unique(s_predict2))}')
        print()
        
    with open(MODEL2_PATH, 'rb') as fp:
        model = pickle.load(fp)
        predict0 = model.predict(train)
        a_predict1 = model.predict(test1)
        a_predict2 = model.predict(test2)

        mse = mean_squared_error(train_y2, predict0)

        print('amount spent')
        print(f'전체 train dataset 에 대한 mse: {mse:.4f}')
        print(f'predict1: {a_predict1}')
        print(f'{len(np.unique(a_predict1))}')
        print()

    print(type(s_predict1))
    print(type(s_predict2))
    print(type(a_predict1))
    print(type(a_predict2))

    print((s_predict1.shape))
    print((s_predict2.shape))
    print((a_predict1.shape))
    print((a_predict2.shape))

    submit_1 = pd.DataFrame({'acc_id':test1_ids, 'survival_time':s_predict1, 'amount_spent':a_predict1})
    submit_2 = pd.DataFrame({'acc_id':test1_ids, 'survival_time':s_predict1, 'amount_spent':a_predict1})
    # submit_1 = pd.DataFrame(data=np.concatenate((test1_ids,s_predict1,a_predict1),axis=1),columns=['acc_id','survival_time','amount_spent'])
    # submit_2 = pd.DataFrame(data=np.concatenate((test2_ids,s_predict2,a_predict1),axis=1),columns=['acc_id','survival_time','amount_spent'])
    # print(sub1.shape)

    # submit_1 = pd.concat([pd.DataFrame({'acc_id':test1_ids}), pd.DataFrame({'survival_time':s_predict1}), pd.DataFrame({'amount_spent':a_predict1})], axis=1)
    # submit_2 = pd.concat([pd.DataFrame({'acc_id':test2_ids}), pd.DataFrame({'survival_time':s_predict2}), pd.DataFrame({'amount_spent':a_predict2})], axis=1)

    # print(submit_1)

    # print('a', len(np.unique(submit_1.acc_id)))
    # print('a', len(np.unique(submit_2.acc_id)))

    # print(submit_1.shape)
    # print(submit_2.shape)

    # print(submit_1.isnull().sum())
    # print(submit_2.isnull().sum())

    submit_1.to_csv(PREDICT1_PATH, index=False)
    submit_2.to_csv(PREDICT2_PATH, index=False)

# 제출
# acc_id, survival_time, amount_spent


# ------------------
#   main
# ------------------
start = time.time()  # 코드 시작 시간

significant_cols = ['acc_id', 'a_playtime', 'a_private_shop', 'a_solo_exp', 'activity_logged_in',
       'amount_spent', 'c_etc_cnt', 'c_num_opponent', 'c_pledge_cnt',
       'changed_lv', 'combat_count', 'combat_logged_in', 'count_sell',
       'max_spent', 'mean_spent', 'median_spent', 'min_spent', 'p_c_char_cnt',
       'p_c_cnt', 'p_etc_cnt', 'p_play_char_cnt', 'p_random_attacker_cnt',
       'p_same_pledge_cnt', 'p_temp_cnt', 'payment_logged_in',
       'pledge_logged_in', 'pledge_num_people', 'qexp_per_playtime',
       'sell_item_amount', 'sell_item_price', 'sexp_per_playtime',
       'tot_c_rank_per_p', 'tot_exp', 'tot_spent', 'tot_trade_amount',
       'total_trade_count', 'trade_logged_in', 'trade_time_bin_0',
       'trade_time_bin_1', 'trade_time_bin_2', 'trade_time_bin_3',
       'trade_type_0']

train_X = pd.read_csv(TRAIN_PATH)
train_X = train_X[significant_cols]

test1 = pd.read_csv(TEST1_PATH)[significant_cols]
test2 = pd.read_csv(TEST2_PATH)[significant_cols]

train_X = preprocess_X(train_X)
train_y = pd.read_csv(LABEL_PATH)  # label에 대한 전처리는 각 함수 내에서 작업.
test1 = preprocess_X(test1)
test2 = preprocess_X(test2)

# scaling
# 전체 train 데이터셋에 대해 fit_transform
# test 데이터셋에 대해 transform
mm = MinMaxScaler()
train_X = mm.fit_transform(train_X)
test1 = mm.transform(test1)
test2 = mm.transform(test2)

create_model_rf(train_X, train_y, size=40000)
test_model_rf(train_X, test1, test2)

exe_time = time.time() - start
print(f'execution time : {exe_time//3600:02.0f}h {exe_time%3600//60:02.0f}m {exe_time%60:02.0f}s')