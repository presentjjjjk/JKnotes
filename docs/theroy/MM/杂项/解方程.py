import numpy as np
from scipy.optimize import fsolve

from math import cos ,radians

# 定义方程组
def equations(vars):
    rho, theta = vars
    R = 1
    theta_ij = 4 * np.pi / 9
    cos_alpha_IO = np.sqrt(3) / 2
    cos_alpha_JO = np.sqrt(3) / 2
    cos_alpha_IJ = cos(radians(60))
    
    eq1 = (rho - R * np.cos(theta)) / np.sqrt(rho**2 - 2 * rho * R * np.cos(theta) + R**2) - cos_alpha_IO
    eq2 = (rho - R * np.cos(theta - theta_ij)) / np.sqrt(rho**2 - 2 * rho * R * np.cos(theta - theta_ij) + R**2) - cos_alpha_JO
    eq3 = (rho**2 - rho * R * np.cos(theta - theta_ij) - rho * R * np.cos(theta) + R**2 * np.cos(theta_ij)) / (
          np.sqrt(rho**2 - 2 * rho * R * np.cos(theta) + R**2) * np.sqrt(rho**2 - 2 * rho * R * np.cos(theta - theta_ij) + R**2)
          ) - cos_alpha_IJ
    
    return [eq1, eq2, eq3]

# 生成大量不同的初始值
initial_guesses = [(rho_guess, theta_guess) for rho_guess in np.linspace(0, 2, 100) for theta_guess in np.linspace(0,  np.pi/2, 100)]

# 求解并筛选非零解
solutions = []
for guess in initial_guesses:
    try:
        solution = fsolve(equations, guess)
        print(f'Initial guess: {guess}, Solution: {solution}')  # 输出初始猜测值和求解结果
        if not np.isclose(solution[0], 0):  # 过滤掉 rho 近似为零的解
            solutions.append(solution)
    except Exception as e:
        print(f'Error with initial guess {guess}: {e}')  # 捕捉并输出异常

# 去重，并输出非零解
unique_solutions = np.unique(np.round(solutions, 6), axis=0)
for sol in unique_solutions:
    print(f'rho: {sol[0]:.6f}, theta: {sol[1]:.6f}')
