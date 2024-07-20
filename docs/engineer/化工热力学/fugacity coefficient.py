import numpy as np
import sympy as sp
from math import sqrt,log
R=8.3145
#CO_2 正丁烷
T_c=[304.19,425.18]
p_c=[7.381,3.797]
w=[0.225,0.193]

T=273.15
p=1.061





#先求独立的系数
a_c=[0,0]
b=[0,0]
a_0=[0,0]
a=[0,0]

for i in range(2):
    a_c[i]=0.457235*(R*T_c[i])**2/p_c[i]
    b[i]=0.077796*R*T_c[i]/p_c[i]
    a_0[i]=(1+(1-(T/T_c[i])**0.5)*(0.37646+1.54226*w[i]-0.26992*w[i]**2))**2
    a[i]=a_c[i]*a_0[i]

#根据得到的系数求总的a,b
#相互作用参数
k=[[0,0.12],[0.12,0]]
x=[0.2,0.8]
y=[0.8962,1-0.8962]


a_t_x=0
a_t_y=0
b_t_x=0
b_t_y=0
for i in range(2):
    b_t_x += x[i]*b[i]
    b_t_y += y[i]*b[i]
    for j in range(2):
        a_ij=sqrt(a[i]*a[j])*(1-k[i][j])
        a_t_x += x[i]*x[j]*a_ij
        a_t_y += y[i]*y[j]*a_ij



#为了计算压缩因子,要求体积根:

#定义PR方程:

V=sp.symbols('V')
pr_1=R*T/(V-b_t_x)-a_t_x/(V*(V+b_t_x)+b_t_x*(V-b_t_x))

pr_2=R*T/(V-b_t_y)-a_t_y/(V*(V+b_t_y)+b_t_y*(V-b_t_y))

f_1=pr_1-p
f_2=pr_2-p

root_x=sp.solve(f_1,V)
root_y=sp.solve(f_2,V)

V_x=float(min([abs(i) for i in root_x]))
V_y=float(max([abs(i) for i in root_y]))

Z_x=p*V_x/(R*T)
Z_y=p*V_y/(R*T)






#计算CO_2的组分逸度系数

#先计算液相x的
t_x=0
t_y=0


for j in range(2):
    a_0j=sqrt(a[0]*a[j])*(1-k[0][j])
    t_x += x[j]*a_0j
    t_y +=y[j]*a_0j

ln_phi_x=b[0]/b_t_x*(Z_x-1)-log(p*(V_x-b_t_x)/(R*T))+a_t_x/(2*sqrt(2)*b_t_x*R*T)*(b[0]/b_t_x-2/a_t_x*t_x)*log((V_x+(1+sqrt(2))*b_t_x)/(V_x-(sqrt(2)-1)*b_t_x))

ln_phi_y=b[0]/b_t_y*(Z_y-1)-log(p*(V_y-b_t_y)/(R*T))+a_t_y/(2*sqrt(2)*b_t_y*R*T)*(b[0]/b_t_y-2/a_t_y*t_y)*log((V_y+(1+sqrt(2))*b_t_y)/(V_y-(sqrt(2)-1)*b_t_y))  

print(f'液相组分逸度系数为:{ln_phi_x:.4f}')
print(f'气相组分逸度系数为:{ln_phi_y:.4f}')