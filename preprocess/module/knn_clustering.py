from .path_header import *

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

import os
from glob import glob


# preprocess.py 실행 후 만들어지는 .csv 파일들 중 최신만 가져온다.
def fetch_data():
	print('\n:::::::: fetch latest dataset...')
	path0 = glob(os.path.join(PREPROCESS_DIR, '*train-v2.csv'))[-1]  # NaN 없는 데이터셋으로 진행
	path1 = glob(os.path.join(PREPROCESS_DIR, '*test1-v2.csv'))[-1]
	path2 = glob(os.path.join(PREPROCESS_DIR, '*test2-v2.csv'))[-1]
	path3 = os.path.join(PREPROCESS_DIR, 'train_label.csv')
	tra = pd.read_csv(path0)
	print(path0)
	te1 = pd.read_csv(path1)
	print(path1)
	te2 = pd.read_csv(path2)
	print(path2)
	lab = pd.read_csv(path3)
	print(path3)
	return tra, te1, te2, lab

# isSurvival, black_cow label 추가
def prepare_label(lab):
	print('\n:::::::: prepare label...')
	## make isSurvival label (categorical)
	lab['isSurvival'] = lab['survival_time'].transform(lambda x: 1 if x==64 else 0)

	## make blac_cow label (categorical)
	quantile = lab.amount_spent.quantile([.25, .5, .7, .8, .9]).tolist()
	# amount_spent 가 상위 10% 이상 유저면 0(heavy), 
	# 상위 10%~30% 유저면 1(middle), 상위 30% 보다 낮으면 2(non-spent) 로 분류하는 함수
	def black_cow(x):  
		if x>quantile[-1]:
			return 0
		elif x>quantile[-3]:
			return 1
		else: 
			return 2
	lab['black_cow'] = lab.amount_spent.transform(black_cow)
	print('isSurvival, black_cow label added')
	return lab

# knn-clustering 후 train,test1,test2 에 merge 할 DataFrame 생성/저장
def knn_clustering():
	tra, te1, te2, lab = fetch_data()
	lab = prepare_label(lab)

	# acc_id 저장
	te1_idx = np.unique(te1.acc_id)
	te2_idx = np.unique(te2.acc_id)

	# 날짜 무시
	tra = tra.drop('day', axis=1).groupby('acc_id').mean()
	te1 = te1.drop('day', axis=1).groupby('acc_id').mean()
	te2 = te2.drop('day', axis=1).groupby('acc_id').mean()

	# robust scaling
	rb = RobustScaler()
	tra_scaled = rb.fit_transform(tra)
	te1_scaled = rb.transform(te1)
	te2_scaled = rb.transform(te2)

	print("\n:::::::: 'isSurvival' clustering...")
	tra_X, val_X, tra_y, val_y = train_test_split(tra_scaled, lab['isSurvival'], train_size=0.8)
	neighbors = 29
	neigh = KNeighborsClassifier(algorithm='auto', n_neighbors=neighbors)
	neigh.fit(tra_X, tra_y)
	score = neigh.score(val_X, val_y)
	print(f'[knn isSurvival]\t{neighbors} neighbors\t score: {score:.4f}')
	te1_predict = neigh.predict(te1_scaled)
	te2_predict = neigh.predict(te2_scaled)
	df1 = pd.DataFrame({'acc_id':te1_idx, 'isSurvival':te1_predict})
	df2 = pd.DataFrame({'acc_id':te2_idx, 'isSurvival':te2_predict})

	print("\n:::::::: 'black cow' clustering...")
	tra_X, val_X, tra_y, val_y = train_test_split(tra_scaled, lab['black_cow'], train_size=0.8)
	neighbors = 29
	neigh = KNeighborsClassifier(algorithm='auto', n_neighbors=neighbors)
	neigh.fit(tra_X, tra_y)
	score = neigh.score(val_X, val_y)
	print(f'[knn black cow]\t\t{neighbors} neighbors\t score: {score:.4f}')
	te1_predict = neigh.predict(te1_scaled)
	te2_predict = neigh.predict(te2_scaled)

	print('\n:::::::: clustered DataFrame saving...')
	te1_predict = pd.DataFrame({'acc_id':te1_idx, 'black_cow':te1_predict})
	te2_predict = pd.DataFrame({'acc_id':te2_idx, 'black_cow':te2_predict})
	res1 = pd.merge(df1, te1_predict, on='acc_id', how='left')
	res2 = pd.merge(df2, te2_predict, on='acc_id', how='left')
	path0 = os.path.join(PREPROCESS_DIR, 'train-knn-clustering.csv')
	path1 = os.path.join(PREPROCESS_DIR, 'test1-knn-clustering.csv')
	path2 = os.path.join(PREPROCESS_DIR, 'test2-knn-clustering.csv')
	lab[['acc_id', 'isSurvival', 'black_cow']].to_csv(path0, index=False)
	res1.to_csv(path1, index=False)
	res2.to_csv(path2, index=False)
	print(f'{path0}\n{path1}\n{path2}\n')