#这个是主文件

#首先,参数输入在主文件中进行

import numpy as np
import sympy as sp
from math import sqrt

from equation_constant import independent_constant,total_constant
from pressure import pressure
from volume_root import volume_root
from yidu_coefficient import yidu_c


T=float(input('请输入温度'))

#定义一个参数数组
T_c=[]
p_c=[]
w=[]



n=int(input('请输入组分个数'))
print('请分别依次输入所有组分的临界温度,临界压力,偏心因子')


for i in range(n):
    t=list(map(float,input().split()))
    T_c.append(t[0])
    p_c.append(t[1])
    w.append(t[2])

'''
T=273.15
n=2
T_c=[304.19,425.18]
p_c=[7.398,3.797]
w=[0.228,0.193]
'''#测试数据,二氧化碳和正丁烷






#输入摩尔分数

x=list(map(float,input('请输入所有组分的液相摩尔分数').split()))

# 获得纯组分的方程常数数组

a,b=independent_constant(T_c,p_c,w,n,T)

# 计算各个组分的饱和蒸气压

p_s=[]

for i in range(n):
    p_s.append(pressure(a[i],b[i],T))

#设定初始的饱和蒸气压

p=sum([x[i]*p_s[i] for i in range(n)])

#设定气相的初始y

y=[x[i]*p_s[i]/p for i in range(n)]


#计算混合物常数

#先给出相互作用参数
k=[]


for i in range(n):
    k.append([])
    for j in range(n):
        k[i].append(float(input('按照顺序输入相互作用参数')))

#测试数据,二氧化碳和正丁烷
#k=[[0,0.12],[0.12,0]]
        
a_l,b_l=total_constant(a,b,x,k,n)

while True:

    V_l=volume_root(p,T,a_l,b_l)[0]

    #计算液相组分逸度系数
    phi_l=[]
    
    

    for i in range(n):
        t_0=0
        for j in range(n):
            a_ij=sqrt(a[i]*a[j])*(1-k[i][j])
            t_0 =t_0+x[j]*a_ij
        phi_l.append(yidu_c(b[i],a_l,b_l,p,V_l,T,t_0))

    
    a_v,b_v=total_constant(a,b,y,k,n)

    #求气相体积根
    V_v=volume_root(p,T,a_v,b_v)[1]

    #求气相组分逸度系数

    phi_v=[]

    for i in range(n):
        t_1=0
        for j in range(n):
            a_ij=sqrt(a[i]*a[j])*(1-k[i][j])
            t_1 =t_1+x[j]*a_ij
        phi_v.append(yidu_c(b[i],a_v,b_v,p,V_v,T,t_1))
    
    #存储一个中间变量
    t=y[:]
    y=[phi_l[i]/phi_v[i]*x[i] for i in range(n)]

    #求两个向量之间的距离

    e_1=sqrt(sum([(y[i]-t[i])**2 for i in range(n)]))

    e_2=abs(sum(y)-1)

    if e_1<1e-6 and e_2<1e-6:
        break
    else:
        p=p*(sum(y))


print('气相组成:')
print(y)

print('系统压力')
print(p)







    




















































