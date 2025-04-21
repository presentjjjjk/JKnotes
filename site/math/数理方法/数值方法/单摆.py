import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from math import radians
# 常数:
g=9.81
l=1

# 定义微分方程

def F(t,Y):
    theta=Y[0]
    omega=Y[1]

    dth_dt=omega
    dom_dt=-g/l*np.sin(theta)

    return [dth_dt,dom_dt]

y0 = [0, radians(10)]

# 时间区间
t_span = [0, 10]  # 根据实际时间区间设置
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 求解微分方程
sol = solve_ivp(F, t_span, y0, t_eval=t_eval)

# 绘图
plt.plot(sol.t, sol.y[0], label='theta(t)')
plt.plot(sol.t, sol.y[1], label='omega(t)')
plt.xlabel('Time t')
plt.ylabel('Solution')
plt.legend()
plt.show()
