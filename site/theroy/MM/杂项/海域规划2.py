import pulp
import matplotlib.pyplot as plt
from math import tan, radians

def fun(n):
    # 定义一些角度常数
    alpha = radians(1.5)
    theta = radians(60)

    # 定义问题
    prob = pulp.LpProblem("line_Problem", pulp.LpMaximize)

    # 定义决策变量x, d
    x = pulp.LpVariable.dicts('x', range(n), lowBound=0, cat='Continuous')
    d = pulp.LpVariable('d', lowBound=0, cat='Continuous')

    D_min = 110 - (7408 - 3704) * tan(alpha)
    D_max = 110 - (0 - 3704) * tan(alpha)

    # 添加约束条件
    prob += 1 - d * 1 / (2 * D_min * tan(theta)) >= 0.1
    prob += 1 - d * 1 / (2 * D_max * tan(theta)) <= 0.2

    prob += x[0] <= D_min * tan(theta)
    prob += x[n - 1] >= 9260 - D_min * tan(theta)

    for i in range(n - 1):
        prob += x[i + 1] == x[i] + d

    for i in range(n):
        prob += x[i] <= 9260

    # 设置合理的目标函数
    prob += d  # 或根据需求优化

    # 求解模型
    return prob.solve(pulp.GUROBI_CMD())  # 使用默认求解器可以改为 pulp.PULP_CBC_CMD()

fun(230)
