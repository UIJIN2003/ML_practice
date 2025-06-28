import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

M = pd.read_csv('C:\\Users\\kim07\\Downloads\\lin_regression_data_01.csv', header=None)  # M에 lin_regression_data
M = M.to_numpy(dtype=float)

x_vector = M[:, 0]
x_dummy = np.array([1]*50)
x_matrix = np.column_stack([x_vector, x_dummy])
# x_matrix = np.append(x_vector, x_dummy, axis = 0)
y_vector = M[:, 1]
y_matrix = np.column_stack([y_vector, y_vector])

alpa = 0.01
step = 50
# =============================================================================
# w0_hist = []
# w1_hist = []
# =============================================================================
MSE_hist = []
w_hist = []
# =============================================================================
# y_hat_matrix = []
# y_hat_matrix = np.array(y_hat_matrix)
# =============================================================================
# =============================================================================
# w_hist = np.array([w_hist])
# MSE_hist = np.array([MSE_hist])
# =============================================================================
# w = np.random.rand(2)*10 


for i in range(step):
    if i == 0:
        w = np.random.rand(2)*10
    w_hist.append(w)
    y_hat = w*x_matrix
    MSE = np.mean((y_hat - y_matrix)**2)
    MSE_hist.append(MSE)
    w_dif = 2*np.mean((y_hat - y_matrix)*x_matrix)
    w = w - alpa*w_dif























# =============================================================================
# w0 = np.random.rand()*10
# w1 = np.random.rand()*10
# 
# w0_hist.append(w0)
# w1_hist.append(w1)
# 
# y_hat = w0*x_vector + w1
# MSE = np.mean((y_hat - y_vector)**2) 
# 
# MSE_hist.append(MSE)
# 
# w0_dif = 2*np.mean((y_hat - y_vector)*x_vector)
# w1_dif = 2*np.mean(y_hat - y_vector)
# 
# w0 = w0 - alpa*w0_dif
# w1 = w1 - alpa*w1_dif
# =============================================================================















# =============================================================================
# w = np.random.rand(2)*10
# w_hist.append(w)
# y_hat = sum((w*x_matrix)[[0,1],:])
# MSE = np.mean((y_hat - y_matrix)**2)
# MSE_hist.append(MSE)
# w_dif = 2*np.mean((y_hat - y_matrix)*x_matrix)
# w = w - alpa*w_dif
# w_hist.append(w)
# y_hat = w*x_matrix
# MSE = np.mean((y_hat - y_matrix)**2)
# MSE_hist.append(MSE)
# w_dif = 2*np.mean((y_hat - y_matrix)*x_matrix)
# w = w - alpa*w_dif
# =============================================================================

# =============================================================================
# 
# for i in range(step):
#     if i == 0: 
#         w = np.random.rand(2)*10
#         
#     w_hist.append(w)
#     y_hat = w*x_matrix
#     w_dif = np.mean((y_hat - y_matrix)*x_matrix*2)
#     MSE = np.mean((y_hat - y_matrix)**2)
#     MSE_hist.append(MSE)
#     w = w - alpa*w_dif
#     
# =============================================================================
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
#     if i == 0:
#         w = np.random.rand(2)*10
#     y_hat = w*x_matrix[i]
#     y_hat_matrix = np.append(y_hat_matrix, y_hat)
#     w_dif = np.mean(y_hat_matrix - y_vector[0:i+1, 0:2])*x_matrix[0:i+1, 0:2]
#     w = w - alpa*w_dif
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
#     if i == 0:
#         y_hat = w*x_matrix[0]
#     else:
#     
#     # y = (w0*x0 + w1*x1)
#         y_hat = w_hist*x_matrix[0:i+1, 0:2]
#     w_dif = np.mean(y_hat - y_vector[0:i+1, 0:2])*x_matrix[:i, 0:2]*2
#     w_hist[i+1] = w_hist[i] - alpa*w_dif
#     MSE = np.mean((y_hat - y_vector[i])**2)
#     w_hist.append(w_hist)
# w_hist = np.array(w_hist)
# =============================================================================
# w_hist = np.append(w_hist, w)
# =============================================================================
#     w_hist = np.append(w_hist[][0], w[0])
#     w_hist = np.append(w_hist[][1], w[1])
#         
# =============================================================================
        # w_hist = np.append(w_hist, w_t)
# =============================================================================
#     MSE = np.mean(sum(y_hat) - y_vector[i])**2
#     np.append(w, w, axis = 0)
#     MSE_matrix[i] = np.column_stack([MSE_matrix, MSE])
#
# =============================================================================

# =============================================================================
#     w0_dif = np.mean(y_hat - y_vector[0])*2*x_vector[0]
#     w1_dif = np.mean(y_hat - y_vector[0])*2
#     w0_new = w0 - alpa*w0_dif
#     w1_new = w1 - alpa*w0_dif
#
# =============================================================================
