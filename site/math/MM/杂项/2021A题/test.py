import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from math import radians

# 使用 'GB2312' 编码读取文件
df = pd.read_csv(r'D:\桌面\HtbJEt9Nb655e46bebfa2a66ec63f940e2da156b\A\附件3.csv', header=0, encoding='GB2312')

# 将每一行转换为列表，并保存到矩阵S中
S = df.values.tolist()
num = len(S)  # 反射镜的个数

# 读取附件1.csv的不同数据
df_1 = pd.read_csv(r'D:\桌面\HtbJEt9Nb655e46bebfa2a66ec63f940e2da156b\A\附件1.csv', header=0, encoding='GB2312')
zuobiao = np.array(df_1.values.tolist())
num_1 = len(zuobiao)  # 主索节点的个数

# 星光的方向向量
alp = radians(36.795)
bet = radians(78.169)

V = np.array([np.cos(bet) * np.cos(alp), np.cos(bet) * np.sin(alp), np.sin(bet)])

zuobiao_1 = []
# 筛选满足条件的主索节点
for i in range(num_1):
    M = np.array([float(zuobiao[i][1]), float(zuobiao[i][2]), float(zuobiao[i][3])])
    if np.linalg.norm(np.cross(M, V)) <= 150:
        zuobiao_1.append(zuobiao[i])

num_2 = len(zuobiao_1)

# 将主索节点的坐标转换为字典，使用字符串作为键
zuobiao_dict = {row[0]: np.array(row[1:], dtype=float) for row in zuobiao_1}

# 初始化坐标，并确保转换为 float 类型
zuobiao_1 = np.array(zuobiao_1)
x = zuobiao_1[:, 1].astype(float)
y = zuobiao_1[:, 2].astype(float)
z = zuobiao_1[:, 3].astype(float)

# 计算 r 的平方根，避免除以零
r = np.sqrt(x**2 + y**2)

# 处理除零情况：r == 0 时，beta 设为 pi/2（垂直向上）或 -pi/2（垂直向下）
beta = np.where(r == 0, np.where(z > 0, np.pi / 2, -np.pi / 2), np.arctan(z / r))

# 方位角，使用 arctan2 处理 x = 0 的情况
alpha = np.arctan2(y, x)

# 书写代价函数
def fun(inp_R):
    # 输入的是半径增量列向量,顺序和zuobiao_1中的顺序一致

    # 更新后的坐标
    dx = inp_R * np.cos(beta) * np.cos(alpha)
    dy = inp_R * np.cos(beta) * np.sin(alpha)
    dz = inp_R * np.sin(beta)

    # 计算更新后的主索节点坐标
    updated_zuobiao = {
        zuobiao_1[i][0]: zuobiao_dict[zuobiao_1[i][0]] + np.array([dx[i], dy[i], dz[i]])
        for i in range(num_2)
    }

    cost = 0  # 初始化代价

    # 遍历反射镜
    for s in S:
        try:
            Q_a = updated_zuobiao[s[0]]
            Q_b = updated_zuobiao[s[1]]
            Q_c = updated_zuobiao[s[2]]
        except KeyError:
            continue  # 如果某个索节点不存在，则跳过

        # 使用向量化计算法向量 n
        n = np.cross(Q_b - Q_a, Q_c - Q_a)
        n /= np.linalg.norm(n)  # 归一化

        # 计算反射光线向量
        M = (Q_a + Q_b + Q_c) / 3
        V_p = np.array([0, 0, -160.2]) - M
        V_p /= np.linalg.norm(V_p)

        # 计算代价函数项
        J = (np.dot(V, n) - np.dot(n, V_p)) ** 2
        cost += J

    return cost

# 定义Optuna优化目标函数
def objective(trial):
    inp_R = [trial.suggest_float(f"r_{i}", -0.6, 0.6) for i in range(num_2)]
    return fun(np.array(inp_R))

# 使用Optuna创建一个研究对象
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

# 输出最优结果
print(f"最优代价值: {study.best_value}")

# 提取优化后的坐标增量
pos = study.best_params.values()
d_x_opt = np.array(list(pos)) * np.cos(beta) * np.cos(alpha)
d_y_opt = np.array(list(pos)) * np.cos(beta) * np.sin(alpha)
d_z_opt = np.array(list(pos)) * np.sin(beta)

# 计算更新后的主索节点坐标
updated_zuobiao = {
    zuobiao_1[i][0]: zuobiao_dict[zuobiao_1[i][0]] + np.array([d_x_opt[i], d_y_opt[i], d_z_opt[i]])
    for i in range(num_2)
}

# 创建DataFrame保存增量和更新后的坐标
columns = ['Node', 'Original_X', 'Original_Y', 'Original_Z', 'Radial_Increment', 'Delta_X', 'Delta_Y', 'Delta_Z', 'Updated_X', 'Updated_Y', 'Updated_Z']
data = []

for i in range(num_2):
    node = zuobiao_1[i][0]
    original_x, original_y, original_z = zuobiao_dict[node]
    radial_increment = pos[i]  # 径向增量
    delta_x, delta_y, delta_z = d_x_opt[i], d_y_opt[i], d_z_opt[i]
    updated_x, updated_y, updated_z = updated_zuobiao[node]
    data.append([node, original_x, original_y, original_z, radial_increment, delta_x, delta_y, delta_z, updated_x, updated_y, updated_z])

# 创建DataFrame并添加径向增量列
df_results = pd.DataFrame(data, columns=columns)

# 保存DataFrame到CSV文件
df_results.to_csv(r'D:\桌面\updated_coordinates1.csv', index=False, encoding='GB2312')

print("坐标增量和更新后的坐标已保存到文件 'updated_coordinates1.csv'.")
