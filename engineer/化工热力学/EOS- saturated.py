import numpy as np
import sympy as sp
from math import log
from math import sqrt

#定义相关常数
R=8.3145

p_c,T_c,w=map(float,input('请依次输入临界压力,临界温度和偏心因子').split())


T=float(input('请输入温度'))

#计算方程常数
a_c=0.457235*(R*T_c)**2/p_c
b=0.077796*R*T_c/p_c
a_0=(1+(1-(T/T_c)**0.5)*(0.37646+1.54226*w-0.26992*w**2))**2
a=a_c*a_0


#给定一个饱和蒸气压的初值
p_s=p_c*10**(7*(1+w)/3*(1-T_c/T))

#求解体积根
V=sp.symbols('V')

#定义PR方程
pr=R*T/(V-b)-a/(V*(V+b)+b*(V-b))

while True:
    f=pr-p_s
    root=sp.solve(f,V)
    root_1 = [abs(i.evalf()) for i in root]

    
    #根据求解得到的体积根计算相关参数
    V_sv=(max(root_1))
    V_sl=(min(root_1))

    Z_sv=p_s*V_sv/(R*T)
    Z_sl=p_s*V_sl/(R*T)

    ln_phi_sv=Z_sv-1-log(p_s*(V_sv-b)/(R*T))-a/(2**1.5*b*R*T)*log((V_sv+(sqrt(2)+1)*b)/(V_sv-(sqrt(2)-1)*b))
    ln_phi_sl=Z_sl-1-log(p_s*(V_sl-b)/(R*T))-a/(2**1.5*b*R*T)*log((V_sl+(sqrt(2)+1)*b)/(V_sl-(sqrt(2)-1)*b))

    print(root_1,p_s,Z_sv,Z_sl,ln_phi_sv,ln_phi_sl)
    #终止条件
    if abs(ln_phi_sv-ln_phi_sl)<1e-5:
        print(f'p_s={p_s:.2f},V_sv={V_sv:.2f},V_sl={V_sl:.2f}')
        break
    else:
        C=(1-(ln_phi_sv-ln_phi_sl)/(Z_sv-Z_sl))
        print(C)
        p_s=p_s*C




    