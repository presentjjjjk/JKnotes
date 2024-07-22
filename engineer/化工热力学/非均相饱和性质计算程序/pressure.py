# 书上的算饱和蒸气压的方法太麻烦了,这里还是用基于麦克斯韦等面积规则和二分法来计算

import sympy as sp
from scipy.integrate import quad


def pressure(a,b,T):
    R=8.314462618
    V=sp.symbols('V')

    #定义pr方程
    pr=R*T/(V-b)-a/(V*(V+b)+b*(V-b))
    
    #求一阶导数找极值点
    y1=sp.diff(pr,V)
    root_2=sp.solve(y1,V)
    root_2=[abs(i) for i in root_2 if abs(i)>b]
    t=[pr.subs({V:i}) for i in root_2]

    #最大和最小对应的p
    p_max=max(t)
    p_min=min(t) if min(t)>0 else 0
    
    p_s=(p_max+p_min)/2 #给定一个有解的饱和蒸气压初值


    while True:
        f_v=R*T/(V-b)-a/(V*(V+b)+b*(V-b))-p_s

        root=sp.solve(f_v,V)
        root=[abs(i) for i in root ]
        if not root:
            print('无解,初值设置不对')
            break
        V_1=min(root)
        V_2=max(root)
        
        #计算定积分

        # 定义数值积分函数
        def f_v_numeric(V):
            return R * T / (V - b) - a / (V * (V + b) + b * (V - b)) - p_s

        # 计算定积分
        x, _ = quad(f_v_numeric, V_1, V_2)

        if abs(x)<=1e-6:    
            break
        else:
            if x>0:
                p_min=(p_max+p_min)/2
                p_s=(p_max+p_min)/2
            else:
                p_max=(p_max+p_min)/2
                p_s=(p_max+p_min)/2
    
    return p_s


    