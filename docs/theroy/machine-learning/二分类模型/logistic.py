import numpy as np

import pandas as pd

from math import sqrt

import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('data.csv')

# 分离特征和标签
X = data[['Feature1', 'Feature2']].values.tolist()  # 将特征转换为列表
y = data['Label'].tolist()  # 将标签转换为列表

#m个训练数据,分别是x=[X[i][0],X[i][1]]

#定义函数:

f=lambda x,w,b:1/(1+np.exp(-(w[0]*x[0]+w[1]*x[1]+b)))

#学习率

a=0.1

#终止条件

e=1e-5

# 系数
w=[1,1]
b=0


while True:

    # 存储上一次的w和b

    w_0=w[:]
    b_0=b

    for j in range(2):
        s=0
        t=0
        for i in range(len(y)):
            x=X[i]
            pL_ipw=(-y[i]*(1-f(w,x,b))+(1-y[i])*f(w,x,b))*x[j]
            s=s+pL_ipw
            t=t+(-y[i]*(1-f(w,x,b))+(1-y[i])*f(w,x,b))
        w[j]=w[j]-a*s/len(y)

    b=b-a*t/len(y)

    if abs(b-b_0)<e and sqrt((w[0]-w_0[0])**2+(w[1]-w_0[1])**2)<e:
        break

w=[float(i) for i in w]
b=float(b)

for i in range(len(w)):
    print(f'系数w{i:}:{w[i]:.3f}')

print(f'系数b:{b:.3f}')

z=[]

for i in range(len(y)):
    
    x=np.array(X[i])
    w=np.array(w)
    z.append(np.dot(w,x)+b)

plt.figure()

plt.scatter(z,y,color='r')

z_0 = np.linspace(min(z)-5, max(z)+5, 40)

y_0=1/(1+np.exp(-z_0))

plt.plot(z_0,y_0,color='g')

plt.show




