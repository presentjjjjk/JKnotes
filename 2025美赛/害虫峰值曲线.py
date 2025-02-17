import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

# 定义系统参数
params = {
    'a1': 0.5,    # 作物基础增长率
    'A': 0.2,     # 季节性波动振幅
    'phi': np.pi/6,     # 相位
    'T': 365,     # 周期（一年）
    'T_AL': 20000, # 温度相关参数
    'T_L': 292,   # 最低适宜温度
    'T_AH': 60000, # 温度相关参数
    'T_H': 303,   # 最高适宜温度
    'K1': 100,    # 作物环境承载力
    'K2': 50,     # 杂草环境承载力
    'alpha': 0.04, # 害虫对作物的捕食系数
    'beta': 0.02,  # 作物对害虫的贡献系数
    'gamma': 0.2783, # 鸟类对害虫的捕食系数
    'mu': 0.01,    # 害虫对鸟类的贡献系数
    'a2': 0.3,    # 杂草增长率
    'b1': 0.005,   # 杂草对作物的竞争系数
    'b2': 0.01,   # 作物对杂草的竞争系数
    'a3': 0.2,    # 害虫自然死亡率
    'a4': 0.05,   # 鸟类自然死亡率
    'k1': 0.01,   # 除草剂效用衰减系数
    'k2': 0.01,   # 杀虫剂效用衰减系数
    'p': 0.1,     # 除草剂对杂草的效用系数
    'q': 0.01,     # 杀虫剂对害虫的效用系数
    'rho1': 0.1,  # 作物的腐殖质生产系数
    'rho2': 0.05, # 杂草的腐殖质生产系数
    'delta': 0.05, # 腐殖质分解系数
    'epsilon': 0.01, # 腐殖质分解的二次项系数
    'phi1': 0.02,  # 腐殖质对作物环境承载力上限的提升系数
    'tau1': 0.01,  # 腐殖质对作物环境承载力上限的提升的半饱和点
    'phi2': 0.005,  # 腐殖质对杂草环境承载力上限的提升系数
    'tau2': 0.00025,  # 腐殖质对杂草环境承载力上限的提升的半饱和点
    'P0': 100,    # 初始除草剂用量
    'Q0': 10,    # 初始杀虫剂用量
    'k': 1,     # 用量衰减因子的速率
    't0': 365,    # 开始快速降低农药用量的时刻
}

def temperature(x):
    w = 2 * np.pi / 365.25  # 基频（一年的周期）
    return 18.055411 + \
           6.491729 * np.cos(w*x) + 1.285268 * np.sin(w*x) + \
           -0.575488 * np.cos(2*w*x) + 0.352078 * np.sin(2*w*x) + \
           0.071302 * np.cos(3*w*x) + 0.021807 * np.sin(3*w*x) + \
           0.201325 * np.cos(4*w*x) +  0.061706 * np.sin(4*w*x)+273.15

# 定义除草剂和杀虫剂的用量函数
def P(t, params):
    return params['P0'] * (1 - 1 / (1 + np.exp(-params['k'] * (t - params['t0']))))

def Q(t, params):
    return params['Q0'] * (1 - 1 / (1 + np.exp(-params['k'] * (t - params['t0']))))

# 定义微分方程组
def system(state, t, params):
    x1, x2, y, z, h = state
    
    # 计算温度响应函数 f_T
    f_T = 1 / (1 + np.exp(params['T_AL']/temperature(t) - params['T_AL']/params['T_L']) 
               + np.exp(params['T_AH']/params['T_H'] - params['T_AH']/temperature(t)))
    
    # 计算季节性变化的生长率 a1'
    a1_prime = params['a1'] + params['A'] * np.cos(2*np.pi*t/params['T'] - params['phi'])
    
    # 计算除草剂和杀虫剂的效用
    P_e = P(t, params) * np.exp(-params['k1'] * t)
    Q_e = Q(t, params) * np.exp(-params['k2'] * t)
    
    # 计算腐殖质对环境承载力的提升
    K1_prime = params['K1'] * (1 + params['phi1'] * h / (1 + params['tau1'] * h))
    K2_prime = params['K2'] * (1 + params['phi2'] * h / (1 + params['tau2'] * h))
    
    # 各个方程
    dx1_dt = a1_prime * f_T * x1 * (1 - x1/K1_prime) - params['alpha']*x1*y - params['b1']*x2 
    dx2_dt = params['a2']*x2*(1 - x2/K2_prime) - min(params['b2']*x1+params['p']*P_e,x2)
    dy_dt = params['beta']*x1*y - min(params['a3']*y+params['gamma']*y*z + params['q']*Q_e,y)
    dz_dt = params['mu']*y*z - min(params['a4']*z,z)
    dh_dt = params['rho1']*x1 + params['rho2']*x2 - params['delta']*h - params['epsilon']*h**2
    
    return [dx1_dt, dx2_dt, dy_dt, dz_dt, dh_dt]

# 设置时间范围和起始年份
n = 50  # 总模拟年数
k = 2  # 从第k年开始绘图
t_max = 365 * n 
t = np.linspace(0, t_max, 100000)

# 计算从第k年开始的索引
start_idx = int((k-1) * (100000/n))

# 设置初始条件
x1_0 = 20    # 作物初始数量
x2_0 = 50     # 杂草初始数量
y_0 = 20     # 害虫初始数量
z_0 = 5      # 鸟类初始数量
h_0 = 10     # 腐殖质初始数量
initial_state = [x1_0, x2_0, y_0, z_0, h_0]

# 求解微分方程组
solution = odeint(system, initial_state, t, args=(params,))


# 计算每年的害虫峰值
def calculate_yearly_pest_peaks(solution, t, start_year=None, end_year=None):
    peaks = []
    years = []
    
    # 每年的数据点数（确保是整数）
    points_per_year = int(len(t) // (t[-1] // 365))
    
    # 如果没有指定起始和结束年份，则使用k和n的值
    if start_year is None:
        start_year = k
    if end_year is None:
        end_year = n
    
    for year in range(start_year-1, end_year):
        # 计算当年的数据范围（使用整数索引）
        start_idx = int(year * points_per_year)
        end_idx = int((year + 1) * points_per_year)
        
        # 获取当年的害虫数量数据
        pest_data = solution[start_idx:end_idx, 2]
        
        # 找出最大值
        peak = np.max(pest_data)
        peaks.append(peak)
        years.append(year + 1)
    
    return years, peaks

# 计算峰值
years, pest_peaks = calculate_yearly_pest_peaks(solution, t)

# ... existing code ...

# 绘制第一个图：年度峰值
plt.figure(figsize=(10, 6))
plt.plot(years, pest_peaks, 'bo-', linewidth=2, markersize=8)
plt.title('Annual Peak Values of Pest Population')
plt.xlabel('Year')
plt.ylabel('Peak Pest Population')
plt.grid(True)

# 计算合适的刻度间隔
total_years = len(years)
if total_years <= 10:
    step = 2
elif total_years <= 20:
    step = 4
elif total_years <= 50:
    step = 10
else:
    step = 20

# 设置横坐标刻度
plt.xticks(years[::step])
plt.show()


