# 这个文件用来计算体积根

import sympy as sp

def volume_root(p,T,a_t,b_t):
    R=8.314462618 
    V=sp.symbols('V')

    #定义pr方程

    pr=R*T/(V-b_t)-a_t/(V*(V+b_t)+b_t*(V-b_t))

    f=pr-p

    root=sp.solve(f,V)

    root=[float(abs(i)) for i in root if abs(i)>b_t]

    V_sl=min(root)

    V_sv=max(root)

    return [V_sl,V_sv]
