# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 간단 실습 1
# =============================================================================
# import numpy as np   #numpy library를 가져오고 np로 선언
#
# A = np.random.rand(10,10)*10    #0에서 10 사이의 랜덤값가지는 10 by 10 행렬을 만들고 A에 넣음
# A_trans = A.transpose()  #np.sum은 행간 계산을 하기때문에 transpose시키고 np.sum을 해 colum 합을 구한다.
# A_row_sum = [0] * 10    # A의 row성분 합을 리스트로 작성할 것, 사이즈가 10이고 원소들은 0.
# A_col_sum = [0] * 10    # A의 col성분 합을 리스트로 작성할 것, 사이즈가 10이고 원소들은 0.
# A_len = len(A)  #A_len에 A의 길이를 저장
# #for문을 써서 행과 열간 합을 각 index에 저장
# for i in range(A_len):      #A list의 길이만큼 반복문을 돌림
#     A_row_sum[i] = np.sum(A[i])     #A행간 합을 A_row_sum에 저장
#     A_col_sum[i] = np.sum(A_trans[i])   #A의 열간 합을 A_col_sum에 저장
#
# A_row_sum_max_index = np.argmax(A_row_sum)   #A_row_sum의 최대값 index를 좌항에 저장
# A_col_sum_min_index = np.argmin(A_col_sum)   #A_col_sum의 최소값 index를 좌항에 저장
#
# #출력
# print("행간 합이 최대가 되는 index는",A_row_sum_max_index, "입니다.")
# print("열간 합이 최소가 되는 index는",A_col_sum_min_index, "입니다.")
#
# =============================================================================

'''보고서 쓸떄 x, y라벨 표시 필수'''

'''신호 분석 하고싶다 하면 주파수의 20배에서 30배 샘플링 해준다'''

''' window shift s ==> 캡쳐'''
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
#
# x_start = -0.1
# x_end = 0.1
# freq = 10
# x_step = 1/(freq*30)  #step이 너무 작아지면 안좋다 => 보기 좋게 적당히 + x_step은 sampling rate
# x_point = np.arange(x_start, x_end + x_step, x_step)
# y_point = np.cos(2*np.pi*freq*x_point)  # y = cos(2 pi ft)
#
# y_point2 = np.cos(2*np.pi*freq*x_point) + 0.5
#
# # y_point = x_point**2 + 1 # x_point의 배열 개수가 21개기 때문에 y_point의 배열도 21개이다/x가 벡터면 y도 벡터
# plt.figure(1)
# # plt.scatter(x_point, y_point)  # scatter는 각 point의 대응되는 점을 찍음
# plt.xlabel("Time[unit]")   #대괄호 안에는 단위
# plt.ylabel("Voltage[unit]")
# plt.grid(True)      #앵간하면 grid 그려주기
# plt.title('1st')
# # plt.legend(['1st', '2nd'])
#
# plt.plot(x_point, y_point,'b-o')#plot은 각 point에 대응되는 점을 직선으로 이음
# plt.figure(2)
# # plt.scatter(x_point, y_point2)
# plt.plot(x_point, y_point2, 'r-^')
# plt.xlabel("Time[unit]")   #대괄호 안에는 단위
# plt.ylabel("Voltage[unit]")
# plt.grid(True)      #앵간하면 grid 그려주기
# plt.title('2nd')
# # plt.legend(['1st', '2nd'])
#
# =============================================================================

# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
#
# count = 0
# count_hist = np.zeros([1,9])
# a = np.arange(1,10,1)
#
# for i in a:
#     count = count + 1
#     count_hist(0, i-1) = count
# # =============================================================================
# =============================================================================
# count_hist = []
# a = np.arange(0 , 10, 1)
# '''for문보다는 직접 대입해보고 하기  ==> 코드가 잘 돌아가는지 확인 하라는 뜻'''
# i = 2
# for i in a:
#     count = count + i
#     count_hist.append(count)
#
# =============================================================================
# append보다 np.zeros를 사용하는거 추천 by 수식쨩


# =============================================================================
# import numpy as np
# 
# monster_HP = 100
# count = 1
# 
# while monster_HP > 0:
#     attack_val = np.round(np.random.rand(1, 1)*4 + 12)  #공격할 떄마다 공격력 바뀜
#     print("몬스터 ", count,"회 공격. 공격력:", attack_val)
#     count += 1
#     monster_HP -= attack_val
#     if monster_HP < 0 :
#         monster_HP = 0
#         
#     print("몬스터에게 상처를 입혔습니다. HP: ", monster_HP, " ")
#     
#     if monster_HP == 0:     #몬스터 죽으면
#         prob_val = np.random.rand(1, 1)
#         if prob_val < 0.01:
#             print("SSS급 무기를 얻었습니다.")
#         elif prob_val < 0.1:
#             print("S급 무기를 얻었습니다.")
#         else:
#             print("B급 무기를 얻었습니다.")
#     
# =============================================================================
    

# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# 
# def uijin_cos(mag_val, freq): #변수 2개를 입력 받을거임 , x1 = 진폭 x2 = 주파수
#     t_start = -0.1
#     t_end = 0.1
#     t_step = 1 / (30*freq)
#     t = np.arange(t_start,t_end,t_step)
# 
#     y1 = mag_val*np.cos(2*np.pi*freq*t)
#     y2 = mag_val*np.cos(2*np.pi*freq*t) + 0.5
#     return y1, y2, t
# 
# mag_val = 2
# freq = 20
# y1, y2, t = uijin_cos(mag_val, freq)
# 
# =============================================================================
# =============================================================================
# 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# x = np.random.rand(3,7)
# save_x = pd.DataFrame(x)
# save_x.to_csv('C:\\Users\\kim07\\.spyder-py3\\Machinlearning_Workplace\\practice.csv', index = False, header = False)
# 
# 
# =============================================================================




























































