import numpy as np
from scipy.optimize import fsolve

# 定义方程组, R = 1
def equations(vars):
    rho, theta = vars
    R = 1  # 固定 R 为 1
    eq1 = np.sqrt(3)/2 - (1 - np.cos(theta - 5*np.pi/9)) / np.sqrt(rho**2 + R**2 - 2*rho*R*np.cos(theta - 5*np.pi/9))
    eq2 = np.sqrt(3)/2 - (1 - np.cos(theta + 5*np.pi/9)) / np.sqrt(rho**2 + R**2 - 2*rho*R*np.cos(theta + 5*np.pi/9))
    return [eq1, eq2]

# 定义更多样化的初始猜测值
initial_guesses = [
    [1, np.pi/4],
    [2, np.pi/3],
    [1, np.pi/6],
    [0.5, np.pi/5],
    [3, np.pi/7],
    [0.1, np.pi/2],
    [5, np.pi/9],
    [10, np.pi/4],
    [0.1, np.pi/3],
    [2, np.pi/4],
    [7, np.pi/3],
    [0.1, np.pi/8],
    [1, np.pi/9],
    [0.01, np.pi/12],
    [4, np.pi/2]
]

solutions = []
for guess in initial_guesses:
    solution = fsolve(equations, guess)
    solutions.append(solution)
    print(f"Initial guess: {guess}, Solution: {solution}")

# 检查解的唯一性
unique_solutions = np.unique(np.round(solutions, decimals=5), axis=0)
print(f"Number of unique solutions: {len(unique_solutions)}")
print(f"Unique solutions: {unique_solutions}")
