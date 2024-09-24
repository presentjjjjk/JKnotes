import numpy as np
from scipy.integrate import solve_ivp, simpson
import pyswarms as ps
import matplotlib.pyplot as plt

def P_mean(k2):
    # 确保 k2 是标量值
    k2 = float(k2)
    k1=80000
    k3=167.8395
    M=2433
    M_p=1165.992
    m=2433
    g=9.8
    rho=1025
    S=np.pi/4
    f=4890
    
    # 定义参数
    a1, a2, a3, a4 = -k2/m, k2/m, -k1/m, k1/m
    b1, b2, b3, b4, b5, omega = k2/(M+M_p), -(k2+k3)/(M+M_p), k1/(M+M_p), -(k1+rho*g*S)/(M+M_p), f/(M+M_p), 2.2143

    # 定义微分方程
    def system(t, y):
        y1, y2, y3, y4 = y
        dy1dt = y3
        dy2dt = y4
        dy3dt = a1 * y3 + a2 * y4 + a3 * y1 + a4 * y2 
        dy4dt = b1 * y3 + b2 * y4 + b3 * y1 + b4 * y2 + b5 * np.cos(omega * t)
        return [dy1dt, dy2dt, dy3dt, dy4dt]

    # 初始条件
    y0 = [0, 0, 0, 0]

    # 时间区间
    t_span = [0, 170]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # 求解微分方程
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

    # 提取速度
    velocities_1 = sol.y[2]
    velocities_2 = sol.y[3]

    # 时间区间在[60, 160]内进行积分
    idx_start = np.searchsorted(t_eval, 60)
    idx_end = np.searchsorted(t_eval, 160)
    t_subset = t_eval[idx_start:idx_end]
    v1_subset = velocities_1[idx_start:idx_end]
    v2_subset = velocities_2[idx_start:idx_end]

    # 计算 k2 * abs(v2 - v1)**2 的积分
    integrand = k2 * np.abs(v2_subset - v1_subset)**2
    integral_value = simpson(integrand, x=t_subset)

    return integral_value/100

# 包装函数，将k2数组映射到每个k2值的P_mean计算
def P_mean_wrapper(k2_array):
    return - np.array([P_mean(k2[0]) for k2 in k2_array])

# 设置k2的搜索范围，例如在[1, 100000]范围内
bounds = (np.array([0]), np.array([100000]))

# 创建PSO优化器
optimizer = ps.single.GlobalBestPSO(n_particles=15, dimensions=1, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

# 使用PSO寻找P_mean的最大值，同时记录每次迭代的代价
cost, pos = optimizer.optimize(P_mean_wrapper, iters=100)

# 绘制代价函数曲线
plt.plot(optimizer.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Value Over Iterations')
plt.grid(True)
plt.show()

# 输出结果
print(f"最优的k2值: {pos[0]}")
print(f"P_mean的最大值: {- cost}")
