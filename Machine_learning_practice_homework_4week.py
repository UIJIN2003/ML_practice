import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''.csv파일 받기'''
M= pd.read_csv('C:\\Users\\kim07\\Downloads\\lin_regression_data_01.csv', header = None) #M에 lin_regression_data
M = M.to_numpy(dtype = float)    #data type이 dataFrame이므로 array of float으로 바꿔준다.

x_point_graph = M[:,0]  #그래프에 그릴 x를 x_point_graph에 받는다.
y_point_graph = M[:,1]  #그래프에 그릴 x를 x_point_graph에 받는다.
x_vector = M[:,0]   # M matrix의 1번째 column성분들을 x_vector에 넣는다.
y_vector = M[:,1]   # M matrix의 2번째 column성분들을 y_vector에 넣는다.


'''Analystic solution으로 w0, w1값 구하기'''
w0 = np.mean((y_vector)*(x_vector - np.mean(x_vector)))/(np.mean(x_vector**2)-(np.mean(x_vector))**2)   #np.mean을 이용해서 1/n*시그마연산을 계산함
w1 = np.mean(y_vector - w0*x_vector)    #구한 w0값을 이용해 w1구함

w0_no_bias = np.mean(x_vector*y_vector)/np.mean(x_vector**2) #bias가 없다고 할 때 w0값

'''loss function (y= w0*x + w1 꼴) 구하기'''
x_start = 0 # x_vector의 최솟값이 5.3이므로 앞으로의 예측을 위해 더 넓게 0으로 설정
x_end = 20  # x_vector의 최댓값이 17.9 이므로 앞으로의 예측을 위해 더 넓게 20으로 설정
x_step = (x_end - x_start) / 50 #우리가 가진 data vector의 size가 50이기 때문에 x범위를 50으로 나눠주어 size를 맞춰줌
x = np.arange(x_start, x_end , x_step)  #x의 값이 일정하게 증가하는 size 50의 벡터 생성
y_hat = w0*x + w1   #단순 선형 조합으로 예측값 y_hat 작성
y_hat_graph = y_hat     #그래프에 넣을 용으로 값 복사해줌

#bias 있을 때와 없을 때 비교하기 위해 bias없는 경우 추가
y_hat_no_bias = w0_no_bias*x            #bias가 없는 함수
y_hat_no_bias_graph = y_hat_no_bias     #그래프 넣을 용으로 값 복사해줌

'''그래프 그리기'''
#rcParams의 기본값을 수정함 (x축, y축 label 크기, 제목크기, 눈금크기, 범례크기)
parameters = {"axes.labelsize": 25, "axes.titlesize": 35, 'xtick.labelsize': 15, "ytick.labelsize": 15, "legend.fontsize": 15}
plt.rcParams.update(parameters)     

#데이터를 그래프에 점으로 보여줌
f, axes = plt.subplots(1, 3)    #그래프 전체 컨트롤, 각 그래프 컨트롤할 수 있는 변수 받음
axes[0].scatter(x_point_graph, y_point_graph)   #외부 파일에 작성된 값을 그래프1에 점으로 찍음
axes[0].set_xlabel("weight [g]")    #그래프1의 x축 이름을 weight [g]로 설정
axes[0].set_ylabel("length [cm]")   #그래프1의 y축 이름을 length [cm]로 설정
axes[0].set_title('data')       #그래프 1의 제목을 data로 설정
axes[0].grid(True)              #그래프 1에 격자 넣음

#데이터를 그래프에 찍으며 Loss function을 그래프에 그림
axes[1].scatter(x_point_graph, y_point_graph)   #그래프2에 data들 점으로 찍음
axes[1].plot(x, y_hat_graph, 'r')   #그래프2에 x에 대응하는 y예측값을 빨간색의 직선으로 이음
axes[1].set_xlabel("weight [g]")    #그래프2의 x축 이름을 weight [g]로 설정
axes[1].set_ylabel("length [cm]")   #그래프2의 y축 이름을 length [cm]로 설정
axes[1].set_title('loss function')  #그래프2의 제목을 loss function으로 설정
axes[1].grid(True)                  #그래프2에 격자를 넣음
axes[1].legend(['data', 'loss function'])   #그래프를 loss function이라고 이름 붙임

#bias가 없는 함수의 Lossfunction
axes[2].scatter(x_point_graph, y_point_graph)   #그래프3에 data들 점으로 찍음
axes[2].plot(x, y_hat_no_bias_graph, 'g')   #그래프3에 x에 대응하는 y예측값을 초록색의 직선으로 이음
axes[2].set_xlabel("weight [g]")    #그래프3의 x축 이름을 weight [g]로 설정
axes[2].set_ylabel("length [cm]")   #그래프3의 y축 이름을 length [cm]로 설정
axes[2].set_title('no bias loss function')  #그래프3의 제목을 loss function으로 설정
axes[2].grid(True)                  #그래프3에 격자를 넣음
axes[2].legend(['data', 'loss function'])   #그래프를 loss function이라고 이름 붙임
plt.show()  #그래프 화면에 띄우기

'''MSE값 구하기'''
MSE = np.mean(y_hat - y_vector)**2  #구한 예측값과 data를 이용해 mean square error를 구함
MSE_no_bias = np.mean(y_hat_no_bias - y_vector)**2  #bias가 없는 경우의 MES
print(MSE)  #MSE 출력
print(MSE_no_bias)  #MSE출력