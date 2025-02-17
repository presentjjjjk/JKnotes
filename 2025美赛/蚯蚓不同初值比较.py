import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

# 复制必要的函数
def temperature(x):
    w = 2 * np.pi / 365.25
    return 18.055411 + \
           6.491729 * np.cos(w*x) + 1.285268 * np.sin(w*x) + \
           -0.575488 * np.cos(2*w*x) + 0.352078 * np.sin(2*w*x) + \
           0.071302 * np.cos(3*w*x) + 0.021807 * np.sin(3*w*x) + \
           0.201325 * np.cos(4*w*x) +  0.061706 * np.sin(4*w*x)+273.15

def P(t, params):
    return params['P0'] * (1 - 1 / (1 + np.exp(-params['k'] * (t - params['t0']))))

def Q(t, params):
    return params['Q0'] * (1 - 1 / (1 + np.exp(-params['k'] * (t - params['t0']))))

def system(state, t, params):
    x1, x2, y, z, h, m = state
    
    f_T = 1 / (1 + np.exp(params['T_AL']/temperature(t) - params['T_AL']/params['T_L']) 
               + np.exp(params['T_AH']/params['T_H'] - params['T_AH']/temperature(t)))
    
    a1_prime = params['a1'] + params['A'] * np.cos(2*np.pi*t/params['T'] - params['phi'])
    
    P_e = P(t, params) * np.exp(-params['k1'] * t)
    Q_e = Q(t, params) * np.exp(-params['k2'] * t)
    
    K1_prime = params['K1'] * (1 + params['phi1'] * h / (1 + params['tau1'] * h))
    K2_prime = params['K2'] * (1 + params['phi2'] * h / (1 + params['tau2'] * h))
    
    dx1_dt = a1_prime * f_T * x1 * (1 - x1/K1_prime) - params['alpha']*x1*y - params['b1']*x2
    dx2_dt = params['a2']*x2*(1 - x2/K2_prime) - min(params['b2']*x1 + params['p']*P_e,x2)
    dy_dt = params['beta']*x1*y - min(params['a3']*y + params['gamma']*y*z + params['q']*Q_e,y)
    dz_dt = params['mu']*y*z - min(params['a4']*z,z)
    dh_dt = (1+np.exp(m))*(params['rho1']*x1 + params['rho2']*x2) - params['delta']*h - params['epsilon']*h**2
    dm_dt = params['kappa']*(x1+x2)*m*(1-m/params['M'])-params['a8']*m
    
    return [dx1_dt, dx2_dt, dy_dt, dz_dt, dh_dt, dm_dt]

# 设置参数
params = {
    'a1': 0.5, 'A': 0.2, 'phi': np.pi/6, 'T': 365,
    'T_AL': 20000, 'T_L': 292, 'T_AH': 60000, 'T_H': 303,
    'K1': 100, 'K2': 50, 'alpha': 0.04, 'beta': 0.02,
    'gamma': 0.3, 'mu': 0.01, 'a2': 0.3, 'b1': 0.005,
    'b2': 0.01, 'a3': 0.2, 'a4': 0.05, 'k1': 0.01,
    'k2': 0.01, 'p': 0.1, 'q': 0.01, 'rho1': 0.1,
    'rho2': 0.05, 'delta': 0.05, 'epsilon': 0.01,
    'phi1': 0.02, 'tau1': 0.01, 'phi2': 0.005,
    'tau2': 0.00025, 'P0': 100, 'Q0': 10, 'k': 1,
    't0': 365, 'kappa': 0.01, 'a8': 0.3, 'M': 5
}

# 设置时间范围
n = 1  # 总模拟年数
t_max = 365 * n 
t = np.linspace(0, t_max, 100000)

# 设置不同的蚯蚓初始值
m0_values = np.linspace(0.01, 10, 100)  # 创建100个不同的初始值

# 固定其他物种的初始值
x1_0 = 32.13    # 作物初始数量
x2_0 = 0        # 杂草初始数量
y_0 = 8.87      # 害虫初始数量
z_0 = 1.42      # 鸟类初始数量
h_0 = 14.76     # 腐殖质初始数量


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