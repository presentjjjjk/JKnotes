import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
    'gamma': 0.3, # 鸟类对害虫的捕食系数
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
    'nu': 0.001,    # 害虫对蝙蝠的贡献系数 (降到最小)
    'xi': 2,    # 蝙蝠对作物的授粉系数 (降到最小)
    'eta': 10,   # 蝙蝠对害虫的捕食系数 (降到最小)
    'a5': 0.005,     # 蝙蝠自然死亡率 (降到最小)
    'xi_half': 0.5,    # 蝙蝠授粉的半饱和常数
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
    x1, x2, y, z, h, w = state
    
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
    dx1_dt = a1_prime * f_T * x1 * (1 - x1/K1_prime) - params['alpha']*x1*y - params['b1']*x2 + params['xi']*x1*w * (params['xi_half']/(params['xi_half'] + w))
    dx2_dt = params['a2']*x2*(1 - x2/K2_prime) - min(params['b2']*x1 + params['p']*P_e,x2)
    dy_dt = params['beta']*x1*y - min(params['a3']*y + params['gamma']*y*z + params['q']*Q_e + params['eta']*y*w,y)
    dz_dt = params['mu']*y*z - min(params['a4']*z,z)
    dh_dt = params['rho1']*x1 + params['rho2']*x2 - params['delta']*h - params['epsilon']*h**2
    dw_dt = params['nu']*y*w - min(params['a5']*w,w)
    
    return [dx1_dt, dx2_dt, dy_dt, dz_dt, dh_dt, dw_dt]

# 设置时间范围和起始年份
n = 5  # 总模拟年数
k = 1  # 从第k年开始绘图
t_max = 365 * n 
t = np.linspace(0, t_max, 100000)

# 计算从第k年开始的索引
start_idx = int((k-1) * (100000/n))


# 作物数量: 32.13
# 杂草数量: -0.00
# 害虫数量: 8.87
# 鸟类数量: 1.42
# 腐殖质数量: 14.76

# 设置初始条件
x1_0 = 32.13    # 作物初始数量
x2_0 = 0    # 杂草初始数量
y_0 = 8.87     # 害虫初始数量
z_0 = 1.42      # 鸟类初始数量
h_0 = 14.76     # 腐殖质初始数量
w_0 = 0.01    # 蝙蝠初始数量（从1降到0.5）
initial_state = [x1_0, x2_0, y_0, z_0, h_0, w_0]

# 求解微分方程组
solution = odeint(system, initial_state, t, args=(params,))

# 在求解微分方程组后，添加以下检验代码
def check_solution(solution, t, params):
    print("开始检验求解结果...")
    
    # 1. 检查是否存在 NaN 或无穷大值
    if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
        print("警告：解中包含 NaN 或无穷大值！")
        return False
    
    # 2. 检查是否存在负值，只显示物种名称
    species = ['作物', '杂草', '害虫', '鸟类', '腐殖质', '蝙蝠']
    has_negative = False
    
    negative_species = []
    for i, name in enumerate(species):
        if np.any(solution[:, i] < 0):
            has_negative = True
            negative_species.append(name)
    
    if has_negative:
        print("\n警告：以下物种出现负值：")
        for name in negative_species:
            print(f"- {name}")
        return False
    
    # 3. 检查解的维度
    expected_shape = (len(t), 6)  # 6个变量：作物、杂草、害虫、鸟类、腐殖质、蝙蝠
    if solution.shape != expected_shape:
        print(f"警告：解的维度不正确！期望维度：{expected_shape}，实际维度：{solution.shape}")
        return False
    
    # 4. 检查系统是否达到相对稳定状态
    # 计算最后1000个时间步的平均值和标准差
    last_period = solution[-1000:, :]
    means = np.mean(last_period, axis=0)
    stds = np.std(last_period, axis=0)
    
    # 计算变异系数（标准差/平均值）
    cv = stds / means
    species = ['作物', '杂草', '害虫', '鸟类', '腐殖质', '蝙蝠']
    
    print("\n系统稳定性分析：")
    for i, name in enumerate(species):
        print(f"{name}的平均值：{means[i]:.2f}，标准差：{stds[i]:.2f}，变异系数：{cv[i]:.2f}")
        if cv[i] > 0.5:  # 设定变异系数阈值为0.5
            print(f"警告：{name}可能未达到稳定状态！")
    
    print("\n检验完成！")
    return True

# 在求解微分方程后调用检验函数
solution_valid = check_solution(solution, t, params)

if solution_valid:
    print("求解结果有效，可以继续进行分析。")
else:
    print("求解结果存在问题，建议检查模型参数和初始条件。")

# 绘图部分
plt.figure(figsize=(15, 10))

# 作物数量变化
plt.subplot(2, 3, 1)
plt.plot(t[start_idx:], solution[start_idx:, 0], 'g-', label='Crop', linewidth=2)
plt.title('Crop Biomass over Time')
plt.xlabel('Time (days)')
plt.ylabel('Biomass')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

# 杂草数量变化
plt.subplot(2, 3, 2)
plt.plot(t[start_idx:], solution[start_idx:, 1], 'r-', label='Weed', linewidth=2)
plt.title('Weed Biomass over Time')
plt.xlabel('Time (days)')
plt.ylabel('Biomass')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

# 害虫数量变化
plt.subplot(2, 3, 3)
plt.plot(t[start_idx:], solution[start_idx:, 2], 'b-', label='Pest', linewidth=2)
plt.title('Pest Density over Time')
plt.xlabel('Time (days)')
plt.ylabel('Density')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

# 鸟类数量变化
plt.subplot(2, 3, 4)
plt.plot(t[start_idx:], solution[start_idx:, 3], 'y-', label='Bird', linewidth=2)
plt.title('Bird Density over Time')
plt.xlabel('Time (days)')
plt.ylabel('Density')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

# 腐殖质数量变化
plt.subplot(2, 3, 5)
plt.plot(t[start_idx:], solution[start_idx:, 4], 'm-', label='Humus', linewidth=2)
plt.title('Humus Amount over Time')
plt.xlabel('Time (days)')
plt.ylabel('Amount')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(t[start_idx:], solution[start_idx:, 5], 'm-', label='Humus', linewidth=2)
plt.title('Bat Density over Time')
plt.xlabel('Time (days)')
plt.ylabel('Amount')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)

plt.tight_layout()
plt.show()


# 绘制相图
plt.figure(figsize=(15, 5))

# 作物-害虫相图
plt.subplot(1, 2, 1)
plt.plot(solution[start_idx:, 0], solution[start_idx:, 2])
plt.title('Crop-Pest Phase Diagram')
plt.xlabel('Crop Biomass')
plt.ylabel('Pest Density')
plt.grid(True)

# 害虫-鸟类相图
plt.subplot(1, 2, 2)
plt.plot(solution[start_idx:, 2], solution[start_idx:, 3])
plt.title('Pest-Bird Phase Diagram')
plt.xlabel('Pest Density')
plt.ylabel('Bird Density')
plt.grid(True)

plt.tight_layout()
plt.show()

# 添加新的图窗，绘制所有生物的数量变化
plt.figure(figsize=(12, 6))
plt.plot(t[start_idx:], solution[start_idx:, 0], 'g-', label='Crop', linewidth=2)
plt.plot(t[start_idx:], solution[start_idx:, 1], 'r-', label='Weed', linewidth=2)
plt.plot(t[start_idx:], solution[start_idx:, 2], 'b-', label='Pest', linewidth=2)
plt.plot(t[start_idx:], solution[start_idx:, 3], 'y-', label='Bird', linewidth=2)
plt.plot(t[start_idx:], solution[start_idx:, 4], 'm-', label='Humus', linewidth=2)
plt.plot(t[start_idx:], solution[start_idx:, 5], 'k-', label='Bat', linewidth=2)
plt.title('Species Population Dynamics in Ecosystem')
plt.xlabel('Time')
plt.ylabel('Population/Density')
plt.legend()
plt.grid(True)
plt.show()

# Add a new figure window specifically for bat population
plt.figure(figsize=(10, 6))
plt.plot(t[start_idx:], solution[start_idx:, 5], 'k-', label='Bat', linewidth=2)
plt.title('Bat Density over Time')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)
plt.show()
