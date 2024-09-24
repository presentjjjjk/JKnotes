import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为SimHei以支持中文字符
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

# 使用 'GB2312' 编码读取文件
df = pd.read_csv(r'D:\桌面\HtbJEt9Nb655e46bebfa2a66ec63f940e2da156b\A\附件3.csv', header=0, encoding='GB2312')

# 将每一行转换为列表，并保存到矩阵S中
S = df.values.tolist()
num = len(S)  # 反射镜的个数

# 读取CSV文件
file_path = r'D:\桌面\updated_coordinates.csv'
df = pd.read_csv(file_path)


# 创建字典：第一列为键，最后三列为值（以元组形式存储）
result_dict = {row[0]: (row[-3], row[-2], row[-1]) for row in df.values}

# 定义一个数组用于存放中心点坐标
M = []

for s in S:
    try:
        Q_a = np.array(result_dict[s[0]])
        Q_b = np.array(result_dict[s[1]])
        Q_c = np.array(result_dict[s[2]])
    except KeyError:
        continue  # 如果某个索节点不存在，则跳过

    # 计算三个点的中心点
    M0 = (Q_a + Q_b + Q_c) / 3
    M.append(M0)

# 将 M 转换为 NumPy 数组
M = np.array(M)

# 拟合旋转抛物面的函数定义
def paraboloid(coords, omega, b):
    x, y = coords
    return omega * (x**2 + y**2) + b

# 提取 M 中的 x, y, z 坐标
x_data = M[:, 0]
y_data = M[:, 1]
z_data = M[:, 2]

# 使用 curve_fit 拟合旋转抛物面参数 omega 和 b
popt, pcov = curve_fit(paraboloid, (x_data, y_data), z_data)

# 提取拟合参数
omega_fit, b_fit = popt
print(f"拟合结果: omega = {omega_fit}, b = {b_fit}")

# 计算代价函数值
z_pred = paraboloid((x_data, y_data), omega_fit, b_fit)
cost_value = np.sum((z_data - z_pred) ** 2)
print(f"代价函数值: {cost_value}")

# 创建网格以绘制拟合的旋转抛物面
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = np.linspace(min(y_data), max(y_data), 100)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)
z_fit = paraboloid((x_fit, y_fit), omega_fit, b_fit)

# 创建图形并设置子图
fig = plt.figure(figsize=(12, 6))

# 左侧子图：绘制原始点和拟合的旋转抛物面
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_data, y_data, z_data, color='r', s=10, label='调整点')  # 点大小为 10
ax1.plot_surface(x_fit, y_fit, z_fit, color='b', alpha=0.5, label='拟合抛物面')

# 设置左侧子图的图例和标题
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('旋转抛物面的拟合')
ax1.legend()


# 右侧子图：绘制 y=0 平面的投影图
ax2 = fig.add_subplot(122)
ax2.scatter(x_data, z_data, color='r', s=10, label='调整点投影')  # 点大小为 10
ax2.plot(x_fit[0], paraboloid((x_fit[0], np.zeros_like(x_fit[0])), omega_fit, b_fit), color='g', label='y=0 平面投影')

# 设置右侧子图的图例和标题
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_title('y = 0 平面的投影')
ax2.legend()


# 显示图形
plt.tight_layout()
plt.show()
