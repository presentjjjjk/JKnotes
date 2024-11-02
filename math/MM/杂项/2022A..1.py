import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义参数
a1, a2, a3, a4 = -10000/2433, 10000/2433, -80000/2433, 80000/2433 # 根据实际情况设置
b1, b2, b3, b4, b5, omega = 10000/(2433+1335.535), -(10000+656.3616)/(2433+1335.535), 80000/(2433+1335.535), -(80000+1025*9.8*np.pi/4)/(2433+1335.535), 6250/(2433+1335.535), 1.4005  # 根据实际情况设置

# 定义微分方程
def system(t, y):
    y1, y2, y3, y4 = y
    dy1dt = y3
    dy2dt = y4
    dy3dt = a1 * y3 + a2 * y4 + a3 * y1 + a4 * y2 
    dy4dt = b1 * y3 + b2 * y4 + b3 * y1 + b4 * y2 + b5 * np.cos(omega * t)
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
output_file_path = r'D:\桌面\displacement_velocity1.xlsx'
df_result.to_excel(output_file_path, index=False)

print(f"表格已成功生成并保存为 {output_file_path}")
