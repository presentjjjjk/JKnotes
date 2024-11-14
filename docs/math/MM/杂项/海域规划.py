import pulp
import matplotlib.pyplot as plt
from math import tan, radians, cos, sin
import numpy as np

# 先给出一个n
n = 33

# 定义一些角度常数
alpha = radians(1.5)
theta = radians(60)

# 定义问题
prob = pulp.LpProblem("line_Problem", pulp.LpMaximize)

# 定义决策变量x, d, D
x = pulp.LpVariable.dicts('x', range(n), lowBound=0, cat='Continuous')
d = pulp.LpVariable.dicts('d', range(n-1), lowBound=0, cat='Continuous')
D = pulp.LpVariable.dicts('D', range(n), lowBound=0, cat='Continuous')

# 添加约束
for i in range(n):
    prob += x[i] <= 7408
    prob += D[i] == 110 - (x[i] - 3704) * tan(alpha)  

for i in range(n-1):
    prob += x[i+1] == x[i] + d[i]
    prob += 0.1 * D[i] <= D[i] - d[i] * cos(theta-alpha) * cos(theta+alpha) / (cos(alpha) * sin(2*theta))
    prob += 0.2 * D[i] >= D[i] - d[i] * cos(theta-alpha) * cos(theta+alpha) / (cos(alpha) * sin(2*theta))

prob += x[0] <= D[0] * cos(alpha) * sin(theta) / cos(alpha+theta)
prob += x[n-1] >= 7408-D[n-1] * cos(alpha) * sin(theta) / cos(theta-alpha)

# 定义 W
W = D[n-1] * cos(alpha) * sin(2*theta) / (cos(theta-alpha) * cos(theta+alpha))

# 添加目标函数
prob += W + cos(theta) / cos(theta+alpha) * (x[n-1] - x[0])

# 求解模型
prob.solve(pulp.GUROBI_CMD())

# 获取解
x_values = [pulp.value(x[i]) for i in range(n)]
D_values = [pulp.value(D[i]) for i in range(n)]

# 画图
plt.figure(figsize=(8, 6))

# 计算并画出x的图像，其中y坐标为斜率为tan(1.5°)的直线对应的x的y坐标加上D值
x_y_values = np.array([(x_val, tan(alpha) * x_val + D_val) for x_val, D_val in zip(x_values, D_values)])

# 使用 plt.scatter 绘制散点图，直接从 x_y_values 数组中提取 x 和 y 的值
plt.scatter(x_y_values[:, 0], x_y_values[:, 1], color='blue', label='x with adjusted y (x * tan(1.5°) + D)')

# 画斜率为tan(1.5度)的直线
x_range = np.arange(0, 8000)
y_line = tan(alpha) * x_range
plt.plot(x_range, y_line, 'r-', label=f'Line with slope tan(1.5°)')

# 绘制从每个蓝色点引出的两条朝下并与坡度直线相交的射线
for x_val, y_val in x_y_values:
    # 计算射线在斜率为tan(1.5°)直线上的交点
    slope = tan(radians(30))
    
    # 第一条射线：从蓝色点向左下方
    x1_end = x_val - (y_val - (tan(alpha) * x_val)) / (slope - tan(alpha))
    y1_end = tan(alpha) * x1_end
    
    # 第二条射线：从蓝色点向右下方
    x2_end = x_val + (y_val - (tan(alpha) * x_val)) / (slope + tan(alpha))
    y2_end = tan(alpha) * x2_end
    
    # 绘制射线
    plt.plot([x_val, x1_end], [y_val, y1_end], 'g-')  # 第一条射线，绿色
    plt.plot([x_val, x2_end], [y_val, y2_end], 'g-')  # 第二条射线，绿色

# 设置图形的范围
plt.xlim(0, 7500)
plt.ylim(0, max([xy[1] for xy in x_y_values]) + 100)

# 添加标签
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x with adjusted y and Line with Slope tan(1.5°)')
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
