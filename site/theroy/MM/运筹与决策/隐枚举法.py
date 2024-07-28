
import numpy as np

# 定义初始可行解

x=np.array([1,0,0])

#定义系数矩阵,目标函数等等

f= lambda x: 3*x[0]-2*x[1]+5*x[2]

A=np.array([[1,2,-1],
            [1,4,1],
            [1,1,0],
            [4,0,1]])
b=[2,4,3,6]

for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            t=np.array([i,j,k])
            if f(t) >= f(x):
                if np.all(np.dot(A,t)<b):
                    x=t

print(f'最优解:{x:}')
print(f'最优值:{f(x):}')