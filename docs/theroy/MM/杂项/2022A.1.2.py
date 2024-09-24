import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# 定义常数
k1 = 80000  # 根据实际情况设置
k3 = 656.3616  # 根据实际情况设置
m = 2433  # 根据实际情况设置
M = 2433   # 根据实际情况设置
M_prime = 1335.535  # 根据实际情况设置
rho = 1025 # 根据实际情况设置
g = 9.8  # 重力加速度
S = np.pi/4  # 根据实际情况设置
f = 6250   # 根据实际情况设置
omega = 1.4005  # 根据实际情况设置

# 定义微分方程组
def system(t, y):
    y1, y2, y3, y4 = y
    k2 = 1000 * np.abs(y4 - y3)**0.5
    dy1dt = y3
    dy2dt = y4
    dy3dt = -k2/m * y3 + k2/m * y4 - k1/m * y1 + k1/m * y2
    dy4dt = k2/(M+M_prime) * y3 - (k2+k3)/(M+M_prime) * y4 + k1/(M+M_prime) * y1 - (k1 + rho*g*S)/(M+M_prime) * y2 + f*np.cos(omega * t)/(M+M_prime)
    return [dy1dt, dy2dt, dy3dt, dy4dt]

# 初始条件
y0 = [0, 0, 0, 0]  # 根据实际初始条件设置

# 时间区间
t_span = [0, 100]  # 根据实际时间区间设置
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 求解微分方程
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# 绘图
plt.plot(sol.t, sol.y[0], label='x1(t)')
plt.plot(sol.t, sol.y[1], label='x2(t)')
plt.xlabel('Time t')
plt.ylabel('Solution')
plt.legend()
plt.show()

# 提取时间点
time_points = [10, 20, 40, 60, 100]  # 在这些时间点采样

# 找到对应时间点的索引
indices = [np.abs(sol.t - t).argmin() for t in time_points]

# 提取位移和速度
displacements = sol.y[0:2, indices]  # 位移 y1 和 y2
velocities = sol.y[2:4, indices]     # 速度 y3 和 y4

# 创建DataFrame
data = {
    'Time (s)': time_points,
    'Displacement of x1': displacements[0],
    'Displacement of x2': displacements[1],
    'Velocity of x1': velocities[0],
    'Velocity of x2': velocities[1],
}

# 创建DataFrame
df_result = pd.DataFrame(data)

# 将结果保存为Excel文件
output_file_path = r'D:\桌面\displacement_velocity2.xlsx'
df_result.to_excel(output_file_path, index=False)

print(f"表格已成功生成并保存为 {output_file_path}")