import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

M= pd.read_csv('C:\\Users\\kim07\\Downloads\\lin_regression_data_01.csv', header = None)
M =M.to_numpy(dtype = float)
# =============================================================================
# M_sorted = np.sort(M, axis = 0) 
# 
# x = M_sorted[:, 0]
# y = M_sorted[:, 1]
# 
# =============================================================================
x_graph = M[:,0]
y_graph = M[:,1]
x_vector = M[:,0]
y_vector = M[:,1]

plt.scatter(x_graph, y_graph)
plt.xlabel("weight [g]")
plt.ylabel("length [cm]")
plt.grid(True)
# plt.legend(['data'])


w0 = np.mean((y_vector)*(x_vector - np.mean(x_vector)))/(np.mean(x_vector**2)-(np.mean(x_vector))**2)
w1 = np.mean(y_vector - w0*x_vector)

x_start = 0
x_end = 20
x_step = 0.4
x = np.arange(x_start, x_end , x_step)
y_hat = w0*x + w1

# plt.scatter(x, y_hat)
plt.plot(x, y_hat, 'r')
plt.xlabel("weight [g]")
plt.ylabel("length [cm]")
plt.grid(True)
plt.legend(['expected line'])

MSE = np.mean(y_hat - y_vector)**2
print(MSE)