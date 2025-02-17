import numpy as np
import itertools
import random as rd

# 定义初始可行解

x=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])



# 定义目标函数

c=np.array([[6,7,11,2],[4,5,9,8],[3,1,10,4],[5,9,8,2]])

f=np.trace(np.dot(c,x))


# 生成所有可能的矩阵的行
all = list(itertools.product([0, 1], repeat=4))

n=0


#选择四个行,把他们组装成矩阵
for x_1 in all:
    for x_2 in all:
        for x_3 in all:
            for x_4 in all:
                x_0=np.array([x_1,x_2,x_3,x_4])
                f_0=np.trace(np.dot(c,np.transpose(x_0)))
                f=np.trace(np.dot(c,np.transpose(x)))
                if f_0 < f:
                    flag=0
                    for i in range(4):
                        if sum([x_0[i][j] for j in range(4)])!=1 or sum([x_0[j][i] for j in range(4)])!=1:
                            flag=1
                    
                    if flag==0:
                        x=x_0
    


print(f'最优解:{x:}')
print(f'最优值:{np.trace(np.dot(c,np.transpose(x))):}')


