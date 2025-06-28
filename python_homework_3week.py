import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

A_matrix = np.zeros([100,100])  #components가 0의 값을 갖는 100 by 100 행렬 A_matrix 생성

#A에 0~99csv파일을 오름차순으로 불러오기 위해 0~99까지 반복하는 for문 생성
for i in range(0, 100, 1):  
    A = pd.read_csv(f'C:\\Users\\kim07\\Desktop\\problem_1_data\\{i}.csv', header = None)   #i번째 데이터를 A에 저장.액셀 파일 첫번째 row를 헤더로 인식해서 header = None 이용
    A = A.to_numpy(dtype = 'float') #DataFrame type에서 float type으로 바꿔줌
    A_matrix[0:100, i] = A[0:100, 25]     #A_matrix의 i번째 colomn index들에 i번째 파일의 26번째 column에 해당하는 components저장
    
plt.imshow(A_matrix, cmap = 'viridis')       #A_matrix 데이터 화면에 시각화, 색상 맵으로 표시
plt.axis('off') #축 표시 off
plt.show()  #그래프 화면에 출력
 
                        
B_matrix = np.zeros([100,100])  #components가 0의 값을 갖는 100 by 100 행렬 B_matrix 생성

#B에 0~99csv파일을 오름차순으로 불러오기 위해 0~99까지 반복하는 for문 생성
for i in range(0, 100, 1):  
    B = pd.read_csv(f'C:\\Users\\kim07\\Desktop\\problem_1_data\\{i}.csv', header = None) # i번째 데이터를 B에 저장
    B = B.to_numpy(dtype = 'float') #DataFrame type에서 float type으로 바꿔줌
    B_matrix[i, 0:100] = B[10, 0:100]     #B_matrix의 i번째 row의 index들에 i번째 파일의 11번째 row에 해당하는 components저장

plt.imshow(B_matrix, cmap = 'viridis')      #B_matrix 데이터 화면에 시각화, 색상 맵으로 표시
plt.axis('off')     #축 표시 off
plt.show()   #그래프 화면에 출력



C_matrix = np.zeros([100,100])  #components가 0의 값을 갖는 100 by 100 행렬 C_matrix 생성

#row와 colomn을 10개씩 짝 지어주기 위해 0~9까지 반복하는 for문 2개 생성
for i in range (0, 10, 1):
    for j in range(0, 10, 1):
        C = pd.read_csv(f'C:\\Users\\kim07\\Desktop\\problem_1_data\\{(i*10+j)}.csv', header = None)    #i*10 + j번째 데이터를 C에 저장
        C = C.to_numpy(dtype = 'float') #DataFrame type에서 float type으로 바꿔줌
        C_matrix[i*10:i*10+10, 10*j:10*j+10] = C[70:80, 80:90]  #이중 반복문을 이용해 i*10~(i*10+10) row, j*10~(j*10+10)에 해당하는 자리에 (i*10+j)번째 파일의 (70:80, 80:90)components저장

plt.imshow(C_matrix, cmap = 'viridis')       #C_matrix 데이터 화면에 시각화, 색상 맵으로 표시
plt.axis('off')     #축 표시 off
plt.show()   #그래프 화면에 출력



 









