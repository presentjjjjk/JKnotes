import numpy as np
import sympy as sp
from math import log
import matplotlib.pyplot as plt

# 定义相关常数
R = 8.3145

# 给定初始值
p_c, T_c, w, T = 3.797, 425.4, 0.193, 273.15

# 计算方程常数
a_c = 0.457235 * (R * T_c)**2 / p_c
b = 0.077796 * R * T_c / p_c
a_0 = (1 + (1 - (T / T_c)**0.5) * (0.37646 + 1.54226 * w - 0.26992 * w**2))**2
a = a_c * a_0

# 给定一个饱和蒸气压的初值
p_s = p_c * 10**(7 * (1 + w) / 3 * (1 - T / T_c))

# 求解体积根
V = sp.symbols('V')

# 定义PR方程
pr = R * T / (V - b) - a / (V * (V + b) + b * (V - b))

# 记录数据以便绘图
ps_values = []
ln_phi_sv_values = []
ln_phi_sl_values = []
V_sv_values = []
V_sl_values = []

while True:
    f = pr - p_s
    root = sp.solve(f, V)
    root_1 = [i for i in root if i.is_real and i > b]

    # 根据求解得到的体积根计算相关参数
    V_sv = max(root_1)
    V_sl = min(root_1)

    Z_sv = p_s * V_sv / (R * T)
    Z_sl = p_s * V_sl / (R * T)

    ln_phi_sv = Z_sv - 1 - log(p_s * (V_sv - b) / (R * T)) - a / (2**1.5 * b * R * T) * log((V_sv + (2**0.5 + 1) * b) / (V_sv - (2**0.5 - 1) * b))
    ln_phi_sl = Z_sl - 1 - log(p_s * (V_sl - b) / (R * T)) - a / (2**1.5 * b * R * T) * log((V_sl + (2**0.5 + 1) * b) / (V_sl - (2**0.5 - 1) * b))

    # 记录数据
    ps_values.append(p_s)
    ln_phi_sv_values.append(ln_phi_sv)
    ln_phi_sl_values.append(ln_phi_sl)
    V_sv_values.append(V_sv)
    V_sl_values.append(V_sl)

    # 终止条件
    if abs(ln_phi_sv - ln_phi_sl) < 1e-5:
        print(f'p_s={p_s:.2f}, V_sv={V_sv:.2f}, V_sl={V_sl:.2f}')
        break
    else:
        C = (1 - (ln_phi_sv - ln_phi_sl) / (Z_sv - Z_sl))
        p_s = p_s * C

# 绘制图像
plt.figure(figsize=(10, 5))

# 绘制PR方程
plt.subplot(1, 2, 1)
V_range = np.linspace(float(b + 1e-5), float(1000), 400)
pr_values = [R * T / (V - b) - a / (V * (V + b) + b * (V - b)) for V in V_range]
plt.plot(V_range, pr_values, label='PR Equation')
plt.scatter(V_sv_values, [p_s] * len(V_sv_values), color='red', label='Saturated Vapor Volume')
plt.scatter(V_sl_values, [p_s] * len(V_sl_values), color='blue', label='Saturated Liquid Volume')
plt.xlabel('Volume (V)')
plt.ylabel('Pressure (P)')
plt.title('PR Equation')
plt.legend()

# 绘制p_s的变化
plt.subplot(1, 2, 2)
plt.plot(ps_values, ln_phi_sv_values, label='ln_phi_sv')
plt.plot(ps_values, ln_phi_sl_values, label='ln_phi_sl')
plt.xlabel('p_s')
plt.ylabel('ln_phi')
plt.title('p_s vs. ln_phi')
plt.legend()

plt.tight_layout()
plt.show()
