import numpy as np
import matplotlib.pyplot as plt

# 初始化缓存字典
x_cache = {}
y_cache = {}

# 定义初始条件
x_cache[(1, 1)] = 0
y_cache[(1, 1)] = 0
x_cache[(2, 1)] = -1/2
y_cache[(2, 1)] = -np.sqrt(3)/2

def x(i, j):
    if (i, j) in x_cache:
        return x_cache[(i, j)]
    
    if j == 1:
        result = x(i-1, 1) - 1/2
    else:
        result = x(i, j-1) + 1
    
    x_cache[(i, j)] = result
    return result
    
def y(i, j):
    if (i, j) in y_cache:
        return y_cache[(i, j)]
    
    if j == 1:
        result = y(i-1, 1) - np.sqrt(3)/2
    else:
        result = y(i, j-1)
    
    y_cache[(i, j)] = result
    return result
    
X = []

for i in range(1, 6):
    for j in range(1, i+1):
        x0 = x(i, j)
        y0 = y(i, j)
        X.append([x0, y0])
        

X = np.array(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.grid(True)
plt.show()
