import numpy as np
import pandas as pd
import os
import cv2

#clustering 이용 함수
def clustering(data):
    
    K = 10                                                                       # 군집 개수
    n, c = data.shape                                                           # data의 row, column 수 n, c에 저장
    
    rand_idx = np.random.choice(n, K, replace=False)                            # n, c값들중 random으로 뽑아 rand_idx에 저장
    m = data[rand_idx].copy()                                                   # data중 rand_idx에 해당하는 값 복사해 중심으로 초기화
    
    while(1): 
        m_prev = m.copy()                                                       # m값 복사해 m_prev에 저장
        
        clus = np.zeros(n)                                                      # 한 데이터의 특징 수가 크기이고 성분 0인 vector 생성
        
        # 한 데이터의 특징 수만큼 반복
        for i in range(n):
            d = (np.sum(((m - data[i]) ** 2), axis = 1)) ** (1 / 2)             #데이터와 각 중심과의 거리 계산(유클리드 거리: norm 2)
            clus[i] = np.argmin(d)                                              #거리중 제일 적은 거리의 index를 clus에 저장
            
        #군집 개수만큼 반복
        for j in range(K):
            clus_p = data[clus == j]                                            #clus데이터중 j군집에 속한 값들 clus_p에 저장
            
            #j군집에 아무것도 없을 경우 방지
            if len(clus_p) > 0:
                m[j] = np.mean(clus_p, axis = 0)                                #j군집의 모든 데이터의 평균을 새로운 군집의 중심으로 저장
        
        #새로운 중심과 이전 중심과 비교(소수점이라 loop 탈출 못할 가능성 존재하여 어느정도 근접하면 멈추게 해줌)
        if np.allclose(m_prev, m):  
            break
    
    return clus, m                                                              # clustering label 데이터랑 중심값 반환

#사과, 복숭아, 토마토 분류 위한 저격 함수
def extract_class_distance(data, labels, clus, m, target_label):                
    labels = np.array(labels, dtype=int)                                        #label numpy array로 변환
    clus = np.array(clus, dtype=int)                                            #clus numpy array로 변환
    max_count = 0
    
    #m 길이만큼 반복
    for k in range(m.shape[0]):
        cluster_labels = labels[clus == k]                                      #clus중 k인 데이터값 모두 저장
        count = np.sum(cluster_labels == target_label)                          #cluster label과 target label이 같은 수를 셈
        if count > max_count:                                                   #count가 max_count보다 클 경우
            class_clus = k                                                      # 그 k값을 목표 class 군집이라고 판단, 저장
            max_count = count                                                   # count 값을 max_count에 옮김
            
    center = m[class_clus]                                                      #그 index를 목표 class 군집의 중심이라 설정
    d = np.log1p(np.sum(((center - data) ** 2), axis = 1)) ** (1 / 2)           #그 중심과 데이터들의 거리 구하기
    d = d.reshape(-1, 1)                                                        #d vector 모양 다듬기

    return d                                                                    # 목표 class 군집과 데이터들 간 거리 벡터 d 반환함

#데이터 정규화 함수
def standard_data (input_data):
    mean = np.mean(input_data, axis=0)                                          #데이터의 평균
    std = np.std(input_data, axis=0)                                            #표준편차
    
    # 표준편차가 0인 경우 방지 => 분모 0되면 오류남
    std[std == 0] = 1
    
    input_data[:, :] = (input_data - mean) / std                                #마지막 label데이터 제외 정규화시킴
    
    return input_data                                                           #input data 반환

#초록 분류
def No1_feature(input_data):
    feature = input_data[:, :, 1].mean()                                        #초록색만 뽑아 평균 취하기, feature에 저장
    
    return feature                                                              #feature 반환

#빨강 분류
def No2_feature(input_data):
    feature = input_data[:, :, 0].mean()                                        #빨간색만 뽑아 평균취하기, featrue에 저장
    
    return feature                                                              #feature 반환

# 가로 분산 => 줄무늬 같은 패턴 분류
def No3_feature(input_data):
    row_mean = np.mean(input_data, axis = (1, 2))                               # 세로줄 평균 취한 것 저장 => 가로방향 밝기 변화
    var = np.var(row_mean)                                                      # 위에서 구한 값 분산 구하기
    
    return var                                                                  # var 반환
    
#청사과, 사과, 복숭아 분류 위함 => 복숭아는 노란색깔 성분 많이 가짐 => 노랑 초록 비율 사과보다 높을 것., 청사과는 빨강 비율 적음
def No4_feature(input_data):
    red_mean = input_data[:, :, 0].mean()                                       #빨간색 밝기 평균
    blue_mean = input_data[:, :, 2].mean()                                      #파란색 밝기 평균
    green_mean = input_data[:, :, 1].mean()                                     #초록색 밝기 평균
    
    feature = (blue_mean + green_mean) / red_mean                               #초록밝기 + 파랑밝기와 빨강밝기 비율 특징으로 저장
    
    return feature                  

#데이터 전처리, 특징추출 함수
def select_features(directory):
    #이미지 파일 directory
    
    
    #이미지 파일 directory 안의 파일 이름들 문자열 리스트로 저장
    file_list = os.listdir(directory)
    
    #특징 저장할 list
    feature_1_list = []
    feature_2_list = []
    feature_3_list = []
    feature_4_list = []
    
    label = []      #정답 라벨
    
    # file_list에 있는 값들 반복
    for name in file_list:
        
        #경로 설정함
        path = os.path.join(directory, name)
    
        #라벨 불러오기(파일명 첫 숫자)
        label.append(int(name.split('_', 1)[0]))
        
        #이미지 읽고 RGB(red, green, blue)값으로 변환하기
        img_GRB = cv2.imread(path)
        img_RGB = cv2.cvtColor(img_GRB, cv2.COLOR_BGR2RGB)
        
        #특징추출하기
        
        # 특징 1: 초록색 분류기
        feature_1 = No1_feature(img_RGB)
        
        # 특징 3: 빨간색 분류기
        feature_2 = No2_feature(img_RGB)
        
        # 가로 전체 밝기 분산 ==> 줄무늬같은 패턴 탐지
        feature_3 = No3_feature(img_RGB)
        
        # 빨간색밝기 평균에 대한 파랑, 초록 밝기 평균 비율
        feature_4 = No4_feature(img_RGB)
    
        #각 리스트에 각특징값 append
        feature_1_list.append(feature_1)
        feature_2_list.append(feature_2)
        feature_3_list.append(feature_3)
        feature_4_list.append(feature_4)
    
    #각 특징 리스트들 numpy array로 변환
    feature_1_list = np.array(feature_1_list)
    feature_2_list = np.array(feature_2_list)
    feature_3_list = np.array(feature_3_list)
    feature_4_list = np.array(feature_4_list)
    
    #features 한군데에 모으기
    features = np.column_stack([feature_1_list, feature_2_list, feature_3_list, feature_4_list])
    
    #clustering 이용
    cluster_features, m = clustering(features)                                  #clustering label, 중심 얻기
    
    #복숭아 군집 중심에 대한 거리 뽑는 함수 
    peach_dist = extract_class_distance(features, label, cluster_features, m, target_label = 6)
    
    #토마토 군집 중심에 대한 거리 뽑는 함수
    tomato_dist = extract_class_distance(features, label, cluster_features, m, target_label = 8)
    
    #복숭아, 토마토, 사과 구분위한 특징까지 포함한 features
    features = np.column_stack([features, peach_dist, tomato_dist])
    
    # features 정규화
    features = standard_data(features)
    
    return features, label                                                      #features와 label 반환