import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
    'K3': 1,     # 浆果灌木环境承载力
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
    'q': 0.1,     # 杀虫剂对害虫的效用系数
    'rho1': 0.1,  # 作物的腐殖质生产系数
    'rho2': 0.05, # 杂草的腐殖质生产系数
    'delta': 0.05, # 腐殖质分解系数
    'epsilon': 0.01, # 腐殖质分解的二次项系数
    'phi1': 0.02,  # 腐殖质对作物环境承载力上限的提升系数
    'tau1': 0.01,  # 腐殖质对作物环境承载力上限的提升的半饱和点
    'phi2': 0.005,  # 腐殖质对杂草环境承载力上限的提升系数
    'tau2': 0.00025,  # 腐殖质对杂草环境承载力上限的提升的半饱和点
    'P0': 1.0,    # 初始除草剂用量
    'Q0': 1.0,    # 初始杀虫剂用量
    'k': 0.1,     # 用量衰减因子的速率
    't0': 365,    # 开始快速降低农药用量的时刻
    'nu': 0.001,    # 害虫对蝙蝠的贡献系数 (降到最小)
    'xi': 2,    # 蝙蝠对作物的授粉系数 (降到最小)
    'eta': 10,   # 蝙蝠对害虫的捕食系数 (降到最小)
    'a5': 0.005,     # 蝙蝠自然死亡率 (降到最小)
    'rho3': 0.05, # 浆果灌木的腐殖质生产系数
    'a6': 0.1,    # 浆果灌木自然增长率
    'a7': 0.1,    # 蛇类自然死亡率
    'sigma': 0.01, # 鸟类对蛇类的贡献系数
    'zeta': 0.01, # 蛇类对鸟类的捕食系数
    'lambda': 0.05, # 鸟类对浆果的捕食系数
    'b3': 0.02,   # 作物对浆果灌木的竞争系数
    'b4': 0.01,   # 杂草对浆果灌木的竞争系数
    'b5': 0.01,   # 浆果灌木对作物的竞争系数
    'b6': 0.01,   # 浆果灌木对杂草的竞争系数
    'b7': 0.01,    # 浆果灌木对鸟类的庇护系数
    'kappa': 0.01,      # 植物对蚯蚓的贡献系数
    'a8':0.3 ,      # 蚯蚓的自然死亡率
    'M': 5   # 蚯蚓的环境承载量
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
    x1, x2, y, z, h, w, x3, u,m = state
    
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
    K3_prime = params['K3'] * (1 + params['phi1'] * h / (1 + params['tau1'] * h))
    
    # 各个方程
    dx1_dt = a1_prime * f_T * x1 * (1 - x1/K1_prime) - min(params['alpha']*x1*y + params['b1']*x2 + params['b5']*x3,x1) + params['xi']*x1*w 
    dx2_dt = params['a2']*x2*(1 - x2/K2_prime) - min(params['b2']*x1 + params['p']*P_e + params['b6']*x3,x2)
    dy_dt = params['beta']*x1*y - min(params['a3']*y + params['gamma']*y*z + params['q']*Q_e + params['eta']*y*w,y)
    dz_dt = params['mu']*y*z - min(params['a4']*z+ params['zeta']*u*z,z)+params['b7']*x3
    dh_dt = (1+(m))*(params['rho1']*x1 + params['rho2']*x2+params['rho3']*x3) - params['delta']*h - params['epsilon']*h**2
    dw_dt = params['nu']*y*w - min(params['a5']*w,w)
    dx3_dt = params['a6']*x3*(1 - x3/K3_prime) -min(params['b3']*x1 + params['b4']*x2,x3-1e-6) + params['lambda']*x3*z
    du_dt = params['sigma']*u*z - min(params['a7']*u,u)
    dm_dt = params['kappa']*(x1+x2)*m*(1-m/params['M'])-params['a8']*m
    
    return [dx1_dt, dx2_dt, dy_dt, dz_dt, dh_dt, dw_dt, dx3_dt, du_dt,dm_dt]

# 设置时间范围和起始年份
n = 3  # 总模拟年数
k = 1  # 从第k年开始绘图
t_max = 365 * n 
t = np.linspace(0, t_max, 1000000)

# 计算从第k年开始的索引
start_idx = int((k-1) * (1000000/n))

# 设置初始条件
x1_0 = 32.13    # 作物初始数量
x2_0 = 0    # 杂草初始数量
y_0 = 8.87     # 害虫初始数量
z_0 = 1.42      # 鸟类初始数量
h_0 = 14.76     # 腐殖质初始数量
w_0 = 0.01    # 蝙蝠初始数量（从1降到0.5）
x3_0 = 0.001     # 浆果灌木初始数量
u_0 = 0.001      # 蛇类初始数量
m_0= 0.01        # 蚯蚓初始数量
initial_state = [x1_0, x2_0, y_0, z_0, h_0, w_0, x3_0, u_0,m_0]

# 修改求解部分
def solve_system(t_span, initial_state, params, method='RK45', max_step=0.1):
    """
    使用更稳健的求解器求解系统
    """
    # 确保初始状态是numpy数组
    initial_state = np.array(initial_state, dtype=float)
    
    # 修改系统函数，确保返回numpy数组
    def system_wrapper(t, state, params):
        return np.array(system(state, t, params), dtype=float)
    
    # 设置求解器参数
    solver_options = {
        'method': method,
        'max_step': max_step,
        'rtol': 1e-3,  # 放宽相对误差容限
        'atol': 1e-6,  # 放宽绝对误差容限
        't_eval': np.linspace(t_span[0], t_span[1], 10000),
        'dense_output': True,  # 启用密集输出
        'vectorized': False
    }
    
    # 求解系统
    try:
        solution = solve_ivp(
            system_wrapper,
            t_span,
            initial_state,
            args=(params,),
            **solver_options
        )
        
        if not solution.success:
            print(f"警告：求解未成功完成！\n原因: {solution.message}")
            
        return solution
        
    except Exception as e:
        print(f"求解过程中出现错误：{str(e)}")
        # 尝试使用备用求解器
        solver_options['method'] = 'RK23'
        print("尝试使用RK23求解器...")
        return solve_ivp(
            system_wrapper,
            t_span,
            initial_state,
            args=(params,),
            **solver_options
        )

# 设置时间范围和求解
t_span = (0, t_max)
solution = solve_system(t_span, initial_state, params)

# 提取结果用于绘图
t = solution.t
solution_array = solution.y.T

# 修改绘图部分
def plot_results(t, solution, start_year=1):
    """
    绘制结果的函数
    """
    start_idx = int((start_year-1) * len(t)/n)
    
    plt.figure(figsize=(15, 10))
    
    species = ['Crops', 'Weeds', 'Pests', 'Birds', 'Humus', 'Bats', 'Berry Shrubs', 'Snakes', 'Earthworm']
    colors = ['g', 'r', 'b', 'y', 'm', 'k', 'c', 'brown', 'orange']
    
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(t[start_idx:], solution[start_idx:, i], colors[i], label=species[i])
        plt.title(f'{species[i]} Over Time')
        plt.xlabel('Time')
        plt.ylabel('Amount')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_in_one(t, solution, start_year=1):
    """
    Plot all species curves in one figure
    """
    start_idx = int((start_year-1) * len(t)/n)
    
    plt.figure(figsize=(15, 8))
    
    species = ['Crops', 'Weeds', 'Pests', 'Birds', 'Humus', 'Bats', 'Berry Shrubs', 'Snakes', 'Earthworms']
    colors = ['g', 'r', 'b', 'y', 'm', 'k', 'c', 'brown', 'orange']
    
    for i in range(9):
        # Smooth the data for clearer visualization
        smoothed_data = savgol_filter(solution[start_idx:, i], 
                                    window_length=101, 
                                    polyorder=3)
        plt.plot(t[start_idx:], smoothed_data, 
                colors[i], 
                label=species[i], 
                linewidth=2)
    
    plt.title('Species Population Dynamics Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Population', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

# Call both plotting functions
plot_results(t, solution_array, start_year=k)
plot_all_in_one(t, solution_array, start_year=k)

# 添加系统状态分析
def analyze_system_stability(solution, t, last_n_points=1000):
    """
    分析系统的稳定性
    """
    species = ['作物', '杂草', '害虫', '鸟类', '腐殖质', '蝙蝠', '浆果灌木', '蛇类', '蚯蚓']
    
    # 分析最后一段时间的数据
    last_period = solution[-last_n_points:, :]
    means = np.mean(last_period, axis=0)
    stds = np.std(last_period, axis=0)
    cv = stds / means  # 变异系数
    
    print("\n系统稳定性分析:")
    print("-" * 50)
    for i, name in enumerate(species):
        print(f"{name}:")
        print(f"  平均值: {means[i]:.4f}")
        print(f"  标准差: {stds[i]:.4f}")
        print(f"  变异系数: {cv[i]:.4f}")
        if cv[i] > 0.1:
            print(f"  警告: {name}可能不稳定")
        print()
    
    return means, stds, cv

# 分析系统稳定性
means, stds, cv = analyze_system_stability(solution_array, t)


