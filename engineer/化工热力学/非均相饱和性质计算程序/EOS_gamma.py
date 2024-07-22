#终于不是二氧化碳和正丁烷了，这次上场的是甲醇和水
from math import log10,exp,log

R=8.314

T_c=[512.58,647.3]
p_c=[8.097,22.064]
w=[0.564,0.344]

alpha=[0.2273,0.2251]

beta=[0.0219,0.0321]

A=[9.4138,9.3876]
B=[3477.9,3826.36]
C=[-40.53,-45.47]

x=[0.4,0.6]

p=0.1013

#估计一个总的临界参数来估计温度初值

T_c_t=x[0]*T_c[0]+x[1]*T_c[1]
p_c_t=x[0]*p_c[0]+x[1]*p_c[1]
w_t=x[0]*w[0]+x[1]*w[1]

T=T_c_t/(1-3*log10(p/p_c_t)/(7*(1+w_t)))

while True:
    T_r=[T/T_c[i] for i in range(2)]
    p_s=[exp(A[i]-B[i]/(C[i]+T)) for i in range(2)]
    V_sl=[(R*T_c[i])/(p_c[i])*(alpha[i]+beta[i]*(1-T_r[i]))**(1+(1-T_r[i])**(2/7)) for i in range(2)]

    #定义模型参数
    Lambda=[[0,0],[0,0]]


    #定义能量参数
    lam=[[0,1085.13],[1631.04,0]]


    for i in range(2):
        for j in range(2):
            if(i==j):
                Lambda[i][j]=1
            else:
                Lambda[i][j]=V_sl[j]/V_sl[i]*exp(-(lam[i][j]-lam[i][i])/(R*T))
    
    #根据wilson模型计算活度系数
    gamma=[]
    
    t_1=-log(x[0]+Lambda[0][1]*x[1])+x[1]*(Lambda[0][1]/(x[0]+Lambda[0][1]*x[1])-Lambda[1][0]/(x[1]+Lambda[1][0]*x[0]))

    gamma.append(exp(t_1))

    t_2=-log(x[1]+Lambda[1][0]*x[0])+x[0]*(Lambda[1][0]/(x[1]+Lambda[1][0]*x[0])-Lambda[0][1]/(x[0]+Lambda[0][1]*x[1]))

    gamma.append(exp(t_2))

    y=[p_s[i]*gamma[i]*x[i]/p for i in range(2)]

    T=T+0.1*(1-sum(y))*T

    if abs(sum(y)-1)<1e-4:
        break

print('气相组成')
print(y)
print('温度')
print(T)

