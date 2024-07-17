#先做一个纯组分的饱和物性求解
import sympy as sp
from scipy.integrate import quad

#p_c,T_c=map(float,input('请输入临界参数').split())
#T_0,w=map(float,input('请输入温度和偏心因子').split())

p_c,T_c,w,T_0=3.797,425.4,0.193,273.15

#p_s=float(input('请给出饱和蒸气压的初值'))
#这个不用了,我在后面已经找到合适的估计饱和蒸气压初值的办法了
p_s=p_c*10**(7*(1+w)/3*(1-T_c/T_0))

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
    #root=[abs(i) for i in root ]
    if not root:
        print('无解,初值设置不对')
        break
    V_1=min(root)
    V_2=max(root)
    #计算定积分

     # 定义数值积分函数
    def f_v_numeric(V):
        return R * T_0 / (V - b) - a / (V * (V + b) + b * (V - b)) - p_s

    # 计算定积分
    x, _ = quad(f_v_numeric, V_1, V_2)

    if abs(x)<=1e-6:
        print(f'饱和蒸气压为{p_s:.4f}')
        print(f'气相饱和体积为{V_1:.4f}')
        print(f'液相饱和体积为{V_2:.4f}')
        break
    else:
        V_0=sum(root)/len(root)#感觉这个迭代条件不是很合适
        p_s=f_v0.subs({V:V_0})
        print(p_s)

