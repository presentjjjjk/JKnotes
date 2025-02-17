import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# 定义系统方程
def system(state, t, params):
    x1, x2, y, z, h, m = state
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = params
    dx1dt = a * x1 - b * x1 * x2
    dx2dt = -c * x2 + d * x1 * x2
    dydt = e * x1 - f * y
    dzdt = g * x1 - h * z
    dhdt = i * x1 - j * h
    dmdt = k * x1 - l * m
    return [dx1dt, dx2dt, dydt, dzdt, dhdt, dmdt]

# 参数
params = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# 初始值
x1_0 = 0.5
x2_0 = 0.5
y_0 = 0.5
z_0 = 0.5
h_0 = 0.5
m_0 = 0.5

# 时间范围
t = np.linspace(0, 100, 1000)

# 创建颜色映射 - 使用plasma色系替代viridis
colors = plt.cm.plasma(np.linspace(0, 1, len(m0_values)))

# 绘制不同初始值的蚯蚓密度变化
fig, ax = plt.subplots(figsize=(12, 8))
for i, m_0 in enumerate(m0_values):
    initial_state = [x1_0, x2_0, y_0, z_0, h_0, m_0]
    solution = odeint(system, initial_state, t, args=(params,))
    
    # 绘制蚯蚓密度随时间的变化
    ax.plot(t, solution[:, 5], color=colors[i], linewidth=2)

# 添加色条 - 使用plasma色系
norm = plt.Normalize(m0_values.min(), m0_values.max())
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
sm.set_array([])
colorbar = plt.colorbar(sm, ax=ax)
colorbar.set_label('Initial Earthworm Density')

ax.set_title('Earthworm Density Evolution with Different Initial Values')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Earthworm Density')
ax.grid(True)
plt.tight_layout()
plt.show() 