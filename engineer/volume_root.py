#先做一个纯组分的饱和物性求解
import sympy as sp

p_c,T_c=map(float,input('请输入临界参数').split())
T_0,w=map(float,input('请输入温度和偏心因子').split())
p_s=float(input('请给出饱和蒸气压的初值'))
#计算相应的参数
R=8.31451
a_c=0.457235*(R*T_c)**2/p_c
b=0.077796*R*T_c/p_c
a_0=(1+(1-(T_0/T_c)**0.5)*(0.37646+1.54226*w-0.26992*w**2))**2
a=a_c*a_0
#定义符号变量和PR方程
V=sp.symbols('V')
f_v0=R*T_0/(V-b)-a/(V*(V+b)+b*(V-b))
while True:
    f_v=R*T_0/(V-b)-a/(V*(V+b)+b*(V-b))-p_s
    root=sp.solve(f_v,V)#注意这里返回的是根的列表
    root=[i for i in root if i.is_real and i>b]
    if not root:
        print('无解,初值设置不对')
        break
    V_1=min(root)
    V_2=max(root)
    #计算定积分
    x=sp.integrate(f_v,(V,V_1,V_2))
    if abs(x)<=1e-6:
        print(f'饱和蒸气压为{p_c:.4f}')
        print(f'气相饱和体积为{V_1:.4f}')
        print(f'液相饱和体积为{V_2:.4f}')
        break
    else:
        V_0=sum(root)/len(root)#感觉这个迭代条件不是很合适
        p_c=f_v0.subs({V:V_0})

