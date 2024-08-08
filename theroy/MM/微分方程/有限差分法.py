import numpy as np
import matplotlib.pyplot as plt


#热传导参数

b=0.5

# 确定步长:

N=20


# 定义一个二维数组,代表函数值

data=np.zeros((N+1,N+1))

for i in range(N+1):
    data[i][N]=100
    data[i][0]=100

for j in range(1,N):
    data[0][j]=0


for n in range(1,N+1):
    for i in range(1,N):
        data[n][i]=data[n-1][i]+b*(data[n-1][i+1]+data[n-1][i-1]-2*data[n-1][i])


# 创建伪彩色图
plt.figure(figsize=(8, 6))
plt.pcolormesh(data, cmap='viridis',shading='auto')
plt.colorbar(label='Value')
plt.title('Pseudocolor Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
