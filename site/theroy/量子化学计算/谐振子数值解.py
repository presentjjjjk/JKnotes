import numpy as np

#定义维度数

N=1000

#把常数k,m,hbar,w全部定义成1

#定义区间和划分区间

a=-10
b=10
x=[]
for i in range(N):
    x.append(a+(b-a)/N*(i+1))

# 定义势能矩阵

V=np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i==j:
            V[i][j]=0.5*x[i]**2

#定义动能矩阵

T=np.zeros((N,N))

dx=(b-a)/N

for i in range(N):
    for j in range(N):
        if i==j:
            T[i][j]=-2/dx**2
        elif j==i+1 or j==i-1:
            T[i][j]=1/dx**2

#哈密顿算子

H=V+(-1/2)*T

# 求解特征值

eigenvalues,eigenvectors= np.linalg.eig(H)

eigenvalues.sort()

print('前几个特征值:')

for i in range(5):
    print(f'{eigenvalues[i]:}')

