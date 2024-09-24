import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 Excel 文件
file_path = r'D:\桌面\Y20WPner9fa62862794e6dc82731a5561ce1132f\B题\附件.xlsx'  # 将 '路径' 替换为实际文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 跳过标题行和列，提取海水深度数据
depth_data = df.iloc[1:, 2:].values

# 现在使用向前差分来估算梯度:
m, n = depth_data.shape

X = np.zeros((m, n))
Y = np.zeros((m, n))

for i in range(m - 1):
    for j in range(n - 1):
        X[i][j] = (depth_data[i][j + 1] - depth_data[i][j]) / 0.02
        Y[i][j] = (depth_data[i + 1][j] - depth_data[i][j]) / 0.02

# 对边界的点采用向后差分估计
for i in range(m):
    X[i][n - 1] = (depth_data[i][n - 1] - depth_data[i][n - 2]) / 0.02
for j in range(n):
    Y[m - 1][j] = (depth_data[m - 1][j] - depth_data[m - 2][j]) / 0.02

# 做一个坐标数组
x = np.linspace(0, 4, n)
y = np.linspace(0, 5, m)

# 定义数据集
inp = []
for i in range(m):
    for j in range(n):
        inp.append(np.array([x[j], y[i], X[i][j], Y[i][j]]))

# 聚类
def cost(K, X):
    model = KMeans(n_clusters=K, random_state=42)
    model.fit(X)
    cost_value = model.inertia_
    return cost_value

# 遍历分类个数
loc = []

for K in range(1, 8):
    loc.append([K, cost(K, inp)])

# 将 loc 转换为 NumPy 数组
loc = np.array(loc)

# 绘制代价函数曲线,确定聚类个数
plt.figure(figsize=(8, 6))
plt.plot(loc[:, 0], loc[:, 1], color='b')
plt.title("The Curve of Cost and K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Cost (Inertia)")
plt.show(block=False)  # 不阻塞，允许继续执行

# 确定最佳聚类数目为K=3，并应用KMeans进行聚类
K_optimal = 3
kmeans_model = KMeans(n_clusters=K_optimal, random_state=42)
kmeans_model.fit(inp)
labels = kmeans_model.labels_

# 将聚类结果转换回二维矩阵的形状，便于绘制
clustered_data = np.reshape(labels, (m, n))

# 绘制聚类结果
plt.figure(figsize=(8, 6))
plt.contourf(x, y, clustered_data, cmap='viridis', alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.title(f"K-Means Clustering Results with K={K_optimal}")
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show(block=False)  # 不阻塞，允许继续执行

# 绘制三维聚类结果图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X_grid, Y_grid = np.meshgrid(x, y)
ax.scatter(X_grid, Y_grid, depth_data, c=clustered_data.flatten(), cmap='viridis')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Depth')
ax.set_title(f"3D K-Means Clustering Results with K={K_optimal}")
plt.show(block=False)  # 不阻塞，允许继续执行

# 为了保证图窗都显示，保持窗口不关闭
plt.show()
