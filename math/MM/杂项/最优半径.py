import numpy as np
import matplotlib.pyplot as plt

# 极坐标矩阵
X = np.array([
    [100., 0.],
    [98., 40.1],
    [112., 80.21],
    [105., 119.75],
    [98., 159.86],
    [112., 199.96],
    [105., 240.47],
    [98., 280.17],
    [112., 320.28]
])

# 阈值
e = 1e-9

# 调整角度坐标
X[:, 1] = np.round(X[:, 1])

# 定义一个最好极径数组:
rho = X[:, 0].copy()

# 定义粒子群优化相关参数
w = 0.9  # 初始惯性权重
c1 = 2.0  # 社会学习因子
c2 = 2.0  # 个体学习因子

# 定义最大速度和最小速度
v_max = 1.0
v_min = -1.0

# 初始化速度
v = np.random.rand(9) * (v_max - v_min) + v_min

def calculate_diff(X, R):
    return (X[:, 0] - R) ** 2

def plot_coordinates(X, R):
    x_coords = X[:, 0] * np.cos(np.radians(X[:, 1]))
    y_coords = X[:, 0] * np.sin(np.radians(X[:, 1]))

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='red')

    # 添加标签
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, str(i), fontsize=12, ha='right')

    # 画圆
    circle = plt.Circle((0, 0), R, color='blue', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    plt.title('Final Coordinates with Optimal Radius Circle')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

max_iterations = 1000
iteration = 0

while iteration < max_iterations:
    # 计算最优半径
    R = np.mean(X[:, 0])

    # 使用 numpy 计算适应度差异
    diff = calculate_diff(X, R)

    # 按照差异排序，并获取最好的两个个体的索引
    sorted_indices = np.argsort(diff)
    I, J = sorted_indices[0], sorted_indices[1]

    # 更新最好极径数组
    better_fit_mask = diff <= (rho - R) ** 2
    rho[better_fit_mask] = X[better_fit_mask, 0]

    # 更新惯性权重（自适应）
    w = 0.9 - 0.5 * (iteration / max_iterations)

    # 更新速度和极径，排除 I 和 J
    for i in range(9):
        if i != I and i != J:
            t1 = (w * v[i] +
                  c1 * np.random.rand() * (X[I, 0] - X[i, 0]) +
                  c2 * np.random.rand() * (rho[i] - X[i, 0]))
            
            # 限制速度
            if t1 >= v_max:
                v[i] = v_max
            elif t1 <= v_min:
                v[i] = v_min
            else:
                v[i] = t1
            
            # 更新极径
            X[i, 0] += v[i]

    # 打印当前速度向量的范数（可选）
    print(f"Iteration {iteration}: Velocity norm = {np.linalg.norm(v)}")

    # 终止条件
    if np.max(diff) <= e:
        break

    iteration += 1

# 最终绘制调整后的坐标
plot_coordinates(X, R)
print('最终半径:', R)
print('坐标:', X)
