import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
    

'''training, validation, test set data 나누는 함수 '''
def data_division(n_data, Tr_rate, V_rate, Te_rate):
    
    np.random.shuffle(n_data)                                               #데이터 섞기
    
    tr_index = int(len(n_data) * Tr_rate / 10)                              #Tr_set 비율만큼 데이터 index 양 확인
    v_index = int(len(n_data) * V_rate / 10)                                #V_set 비율만큼 데이터 index 양 확인
    te_index = int(len(n_data) * Te_rate / 10)                              #Te_set 비율만큼 데이터 index 양 확인
    
    #비율대로 data 나누기
    tr_set = n_data[0:tr_index]                                             
    v_set = n_data[tr_index : tr_index + v_index]
    te_set = n_data[tr_index + v_index : tr_index + v_index + te_index]
    
    return tr_set, v_set, te_set

'''받아온 파일의 데이터 x와 y로 자동 분류 해주는 함수'''
def make_input_output(M):
    
    for i in range(M.shape[1]):                                               #M의 column 수만큼 반복
        if i == 0:
            x_matrix = M[:, 0]                                                  #M의 첫번째 column 성분 값들 x_matrix에 저장
        elif i < (M.shape[1] - 1):
            x_matrix = np.column_stack([x_matrix, M[:, i]])                     #M의 마지막 column 성분 제외한 값들 x_matrix에 저장
        else:
            y = M[:, i]                                                         #M의 마지막 column 성분 y에 저장
    y = y.reshape(y.shape[0], 1)                                                   #y size 다듬기
    x_matrix_t = np.transpose(x_matrix)                                         #한 데이터에 대한 특징들 한 column에 나타나기 위해 transpose 
    y_t = np.transpose(y)                                                       #위와 동일
    return x_matrix_t, y_t

'''데이터의 class수 세는 함수'''
def y_class(y):
    
    y_class = np.unique(y)                                                      #class 수 계산
    Q = len(y_class)                                                            #numpy array로 받아지기 때문에 길이를 셈
    
    return Q

'''데이터의 특징 수 세는 함수'''
def ch_count(y):
    
    Q = len(y)                                                                  #받아온 데이터의 열 개수를 셈
    
    return Q

'''One-Hot Encoding 구현 함수'''
def One_Hot_Encoding(y):
    Q = y_class(y)                # 예: Q=3이면
    y_vector = np.zeros((Q, y.shape[1]))
    for i in range(y.shape[1]):
        label = int(y[0, i])      # y는 정수로 가정
        y_vector[label, i] = 1
    
    return y_vector

'''row기반 dummy추가해주는 함수'''
def add_dummy(x):
    
    x_dummy = np.ones(x.shape[1])                                               #입력데이터 x의 길이만큼 dummy 생성
    x = np.row_stack([x, x_dummy])                                              #row방향으로 쌓음
    
    return x

'''sigmoid 구현 함수'''
def sigmoid_function(z):
    
    return(1/(1 + np.exp(-z)))   

    
'''대표값 찾아서 1로 만들어주는 함수'''
def classification_data_max(y):
    
    p = np.zeros_like(y)                                                        #받아온 y데이터의 row와 column 크기만큼 요소가 0인 matrix 생성
    
    y_max = np.argmax(y, axis = 0)                                              #y의 최댓값 index 저장
    
    #y 데이터 길이만큼 p (i번째 최댓값 index, i)에 1 저장
    for i in range(y.shape[1]):
        p[y_max[i], i] = 1
                
    return p


'''데이터 정확도 함수'''
def data_accuracy(y_h, y):                                                      #한 데이터에 대한 성분 row로 나열한 데이터기준
    
    count = 0                                                                   #count 기능 이용할 변수 0으로 초기화
    
    for i in range(y_h.shape[1]):                                             #예측 데이터 column 성분만큼 반복
        if (y_h[:, i] == y[:, i]).all():                                        #받아온 데이터와 예측 데이터의 같은 column의 row성분 값이 모두 같은지 확인
            count += 1                                                          #위 조건에 해당할 때 count
            
    accuracy = count / y_h.shape[1]                                           #count 된 수를 예측데이터 column 개수만큼 나눠줌
    
    return accuracy


'''batch size 1 forward_propagation 구현 함수'''
def forward_propagation_1(x_input_added_dummy, v_matrix, w_matrix, L):          #Hidden Layer의 node 수 지정

    alpha = np.dot(v_matrix, x_input_added_dummy)                               #v와 xinput을 곱해 alpha를 구함
    b_matrix = sigmoid_function(alpha).reshape(-1, 1)                           # batch size 1일 때 sigmoid에 넣으면 형태 깨져서 reshape이용
   
    b_matrix = add_dummy(b_matrix)                                              #b에 dummy 추가
                                      
    beta = np.dot(w_matrix, b_matrix)                                           #w와 b 곱해서 beta 구함
    y_hat = sigmoid_function(beta)                                              #beta를 sigmoid function에 넣어 y_hat 구함
    
    return y_hat, b_matrix   


'''batch size 1 back propagation 구현 함수'''
def Back_Propagation_1(y_hat, y_data, x_matrix_added_dummy, b_matrix, w_prev, L): # w먼저 weight update시키므로 update 전 w 입력 받음
    
    # w 기울기 구하는 코드
    delta = 2 * (y_hat - y_data.reshape(-1, 1)) * y_hat * (1 - y_hat)           #delta 구함, y_data는 (:, 1)로 슬라이스 된 크기
    
    w_dif = np.dot(delta, b_matrix.T)                                           #delta와 b를 이용해 w의 기울기 구함
    
    # v 기울기 구하는 코드
    proc = np.dot(delta.T, w_prev) 

    #dummy data 삭제                                             
    b_matrix_h = np.delete(b_matrix, L, axis = 0)                                 
    proc = np.delete(proc, L, axis = 1 )                                        
    
     
    v_dif = np.dot((proc.T * b_matrix_h * (1 - b_matrix_h)), x_matrix_added_dummy.reshape(1, -1))     # v의 기울기 구하기
    
    return w_dif, v_dif                                                         # 함수의 반환값으로 w와 v의 기울기를 반환함


'''batch size 1인 Two_Layer_Neural Network'''
def Two_Layer_Neural_Network_1(x_input, y_data, L, epoch, LR):    
                                                               
    MSE_list = []                                                               #MSE 저장할 list
    ACCURACY_list = []                                                          #accuracy 저장할 list
    x_matrix = add_dummy(x_input)                                               #입력에 dummy data 추가
    
    M = ch_count(x_input)                                                       #input 속성 수 체크
    Q = ch_count(y_data)                                                        #ouput class 수 체크
    
    # weight 초기화
    v = np.random.rand(L, M + 1) * 2 - 1 
    w = np.random.rand(Q, L + 1) * 2 - 1 
    
    # epoch수 만큼 반복
    for i in range(epoch):
        
        y_hat_all_epoch = []                                                    #한 epoch마다 y_hat 저장하는 list 초기화
        
        #데이터 길이만큼 반복
        for j in range(y_data.shape[1]):
            w_prev = w.copy()                                                   #update전 weight값 저장
            
            y_hat, b_matrix = forward_propagation_1(x_matrix[:, j], v, w, L)    #forward propagation 진행
            y_hat_all_epoch.append(y_hat)                                       #y_hat 값 list에 저장
            
            w_dif, v_dif = Back_Propagation_1(y_hat, y_data[:, j], x_matrix[:, j], b_matrix, w_prev, L)    #back propagation 진행

            #weight update
            w = w - LR * w_dif
            v = v - LR * v_dif
            
        y_hat_all = np.hstack(y_hat_all_epoch)                                  #y_hat을 쌓은 list에 numpy array를 배열로 만들어줌
        error = y_hat_all - y_data                                              #error 계산
        MSE = np.mean(error ** 2)                                               #MSE 계산
        MSE_list.append(MSE)                                                    #MSE list에 저장
        
        P = classification_data_max(y_hat_all)                                  #데이터 당 최댓값을 1로 만들어주는 분류 함
        accuracy = data_accuracy(P, y_data)                                     #accuracy 구하기
        ACCURACY_list.append(accuracy)                                          #accuracy list에 저장
        
    return MSE_list, ACCURACY_list, v, w                                        #MSE_list, ACCURACY_list, v, w 반환함

'''confusion matrix 구현 함수'''
def confusion_matrix(y_hat, y_data):
    
    y_pred_index = np.argmax(y_hat, axis = 0)                                   #y_hat 데이터당 최댓값 index 가져옴
    y_true_index = np.argmax(y_data, axis = 0)                                  #y_data 데이터당 최댓값 index 가져옴
    
    true_num = 0                                                                #정확히 예측한 횟수 초기화
    
    classes_num = ch_count(y_data)                                              #y_data class 수 체크
    
    confusion_matrix = np.zeros((classes_num + 1, classes_num + 1))             #정확도 나타내기 위해 class수 + 1개만큼 정방 행렬 만듦
    
    #y 길이만큼반복
    for i in range(len(y_pred_index)):
        confusion_matrix[y_true_index[i], y_pred_index[i]] +=  1                #실제값, 예측값 index에 해당하는 자리에 1 더함
        
    # class 수만큼 반복
    for i in range(classes_num):
        
        # row방향으로 더한 값이 0보다 클 때 전체 데이터로 정확히 예측한 값 나눠줌
        if sum(confusion_matrix[i, : classes_num]) > 0:
            confusion_matrix[i, classes_num] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, : classes_num])
        
        # column 방향으로 더한 값이 0보다 클 때 전체 데이터로 정확히 예측한 값 나눠줌
        if sum(confusion_matrix[: classes_num, i]) > 0:
            confusion_matrix[classes_num, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[: classes_num, i])
        
        true_num += confusion_matrix[i, i]                                      #정확히 예측한 값 세기
    confusion_matrix[classes_num, classes_num] = true_num / len(y_pred_index)   #전체 데이터에 대한 정확도 마지막 index에 저장
    
    return confusion_matrix                                                     #confusion_matrix 반환



def select_features(directory):
    #이미지 파일 directory
    
    
    #이미지 파일 directory 안의 파일 이름들 문자열 리스트로 저장
    file_list = os.listdir(directory)
    
    #첫번째 특징 저장할 list
    feature_1_list = []
   
    
    label = []      #정답 라벨
    
    for name in file_list:
        
        #경로 설정함
        path = os.path.join(directory, name)
    
        #라벨 불러오기(파일명 첫 숫자)
        label.append(int(name.split('_', 1)[0]))
        
        #이미지 읽고 RGB(red, green, blue)값으로 변환하기
        img_GRB = cv2.imread(path)
        img_RGB = cv2.cvtColor(img_GRB, cv2.COLOR_BGR2RGB)
        
        #특징추출하기
        
        # 특징 1: 전체 image pixel 값의 평균을 냄 ==> 색깔 나눔
        feature_1 = histogram_feature(img_RGB)
    
        feature_1_list.append(feature_1)
       
    
    # feature_1_list = np.array(feature_1_list)
    features = np.array(feature_1_list)
 
    
    # features = np.column_stack([feature_2_list, feature_3_list, feature_5_list, feature_6_list, feature_7_list])
    # features = standard_data(features)
    
    return features, label

def standard_data (input_data):
    feature_data = input_data[:, :-1]  # 마지막 열(label)을 제외한 feature만 추출
    
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    
    # 표준편차가 0인 경우(=변화 없는 feature)는 나누지 않도록 처리
    std[std == 0] = 1
    
    input_data[:, :-1] = (feature_data - mean) / std
    
    return input_data

#초록 분류
def No2_feature(input_data):
    feature = input_data[:, 0, :].mean()
    
    return feature

#빨강 분류
def No3_feature(input_data):
    feature = input_data[0, :, :].mean()
    
    return feature

#가로 평균
def No4_feature(input_data):
    row_mean = np.mean(input_data, axis = (1, 2))
    mean = np.mean(row_mean)
    
    return mean

# 가로 분산 => 줄무늬 같은 패턴 분류
def No5_feature(input_data):
    row_mean = np.mean(input_data, axis = (1, 2))
    var = np.var(row_mean)
    
    return var

#대각성분 밝기 평균
def No6_feature(input_data):
    all_mean = np.mean(input_data, axis = 2)
    diagonal_components = np.diagonal(all_mean)
    mean = np.mean(diagonal_components)
    
    return mean
    
#청사과, 사과, 복숭아 분류 위함 => 복숭아는 노란색깔 성분 많이 가짐 => 노랑 초록 비율 사과보다 높을 것., 청사과는 빨강 비율 적음
def No7_feature(input_data):
    red_mean = input_data[0, :, :].mean()
    blue_mean = input_data[:, :, 0].mean()
    green_mean = input_data[:, 0, :].mean()
    
    feature = (blue_mean + green_mean) / red_mean
    
    return feature

def histogram_feature(img):
    r_hist = np.histogram(img[:, :, 0], bins=16, range=(0, 256))[0]
    g_hist = np.histogram(img[:, :, 1], bins=16, range=(0, 256))[0]
    b_hist = np.histogram(img[:, :, 2], bins=16, range=(0, 256))[0]
    hist = np.concatenate([r_hist, g_hist, b_hist])
    return hist / np.sum(hist)

directory = "C:\\Users\\kim07\\Desktop\\Machinlearning_Workplace\\train"
L = 25
LR = 0.01
epoch = 1000
Tr_rate = 7
Val_rate = 3 
Te_rate = 0
K = 10


data, label = select_features(directory)

n, c = data.shape

# m = np.random.rand(K, c) * 4 - 2

rand_idx = np.random.choice(n, K, replace=False)
m = data[rand_idx].copy()

while(1): 
    m_prev = m.copy()
    
    clus = np.zeros(n)
    for i in range(n):
        d = (np.sum(((m - data[i]) ** 2), axis = 1)) ** (1 / 2)
        clus[i] = np.argmin(d)
        
        
    for j in range(K):
        clus_p = data[clus == j]
        if len(clus_p) > 0:
            m[j] = np.mean(clus_p, axis = 0)
    
    if np.allclose(m_prev, m):
        break
    
# cluster_data = np.column_stack([data, ])

# =============================================================================
# from sklearn.decomposition import PCA  # row-level 조건이 없으면 OK
# 
# # 2차원으로 줄이기 (시각화용)
# pca = PCA(n_components=2)
# data_2d = pca.fit_transform(data)
# 
# # 군집 결과 색깔로 scatter plot
# plt.figure(figsize=(8, 6))
# for k in range(K):  # 군집 수만큼
#     cluster_points = data_2d[clus == k]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k}', alpha=0.7)
# 
# plt.title("KMeans Clustering Visualization (PCA)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# =============================================================================

total_data = np.column_stack([data, clus])

total_data = standard_data(total_data)

tr_set, val_set, te_set = data_division(total_data, Tr_rate, Val_rate, Te_rate)

x_features, y_label = make_input_output(tr_set)
x_features_val, y_label_val = make_input_output(val_set)

y_data = One_Hot_Encoding(y_label)
y_data_val = One_Hot_Encoding(y_label_val)

MSE_tr, Accuracy_tr, v_tr, w_tr = Two_Layer_Neural_Network_1(x_features, y_data, L, epoch, LR)

y_hat_val_all = []                                                             #test set의 y_hat 저장할 list

x_features_val = add_dummy(x_features_val)                                                      #dummy 추가

# training set으로 학습한 weight로 test set forwoard propagation
#batch size 1이므로 데이터 하나씩 forward propagation
for i in range(x_features_val.shape[1]): 
    y_hat_val, _ = forward_propagation_1(x_features_val[:, i], v_tr, w_tr, L)       #forward propagation 진행
    y_hat_val_all.append(y_hat_val)                                               #y_hat_te list에 저장

y_hat_val = np.hstack(y_hat_val_all)


#test set에 대한 confusion matrix
confusion_matrix_val = confusion_matrix(y_hat_val, y_data_val)









  








from sklearn.metrics import confusion_matrix
 
true_label = np.array(label)         # 진짜 정답
cluster_label = clus.astype(int)     # 군집 번호

cm = confusion_matrix(true_label, cluster_label)
print(cm)



























parameters = {"axes.labelsize": 20, "axes.titlesize": 30, 'xtick.labelsize': 12, "ytick.labelsize": 12, "legend.fontsize": 12}
plt.rcParams.update(parameters) 

