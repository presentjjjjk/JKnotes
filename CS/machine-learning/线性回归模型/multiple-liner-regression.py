import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一系列三维的点
x_1 = np.random.rand(20) + 10
x_2 = np.random.rand(20) + 5
x = np.array([x_1, x_2])

y = np.random.rand(20) + 1

a = 0.001  # 学习率
tolerance = 1e-3  # 迭代终止条件

# 设定初始的w和b
w_1 = 0
w_2 = 0
w = np.array([w_1, w_2])
b = 0

# 迭代更新 w 和 b
while True:
    y_pred = np.dot(w, x) + b
    error = y_pred - y
    decrease_1 = np.dot(error, x_1) / 20
    decrease_2 = np.dot(error, x_2) / 20
    decrease_3 = np.sum(error) / 20

    w_1 = w_1 - a * decrease_1
    w_2 = w_2 - a * decrease_2
    b = b - a * decrease_3
    w=np.array([w_1, w_2])

    if np.abs(decrease_1) < tolerance and np.abs(decrease_2) < tolerance and np.abs(decrease_3) < tolerance:
        break

print(f'w_1={w_1:.2f}, w_2={w_2:.2f}, b={b:.2f}')

# 绘制三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(x_1, x_2, y, c='r', marker='o')

# 生成网格点
x1, x2 = np.meshgrid(np.linspace(9, 11, 40), np.linspace(4, 6, 40))
y_0 = w_1 * x1 + w_2 * x2 + b

# 绘制拟合平面
ax.plot_surface(x1, x2, y_0, alpha=0.5, rstride=100, cstride=100, color='blue')

ax.set_xlabel('X1 axis')
ax.set_ylabel('X2 axis')
ax.set_zlabel('Y axis')

plt.show()
