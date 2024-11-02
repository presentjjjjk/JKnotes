import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 设置随机种子
np.random.seed(42)

# 生成4个簇:
cluster_1 = np.random.normal(loc=[4.5, 5], scale=0.5, size=(10, 2))
cluster_2 = np.random.normal(loc=[9, 4], scale=0.6, size=(10, 2))
cluster_3 = np.random.normal(loc=[7, 6], scale=0.55, size=(10, 2))
cluster_4 = np.random.normal(loc=[8, 8], scale=0.73, size=(10, 2))

# 拼接数据集:
X = np.vstack((cluster_1, cluster_2, cluster_3, cluster_4))

# 设置分类个数
K = 4

# 创建KMeans模型
model = KMeans(n_clusters=K, random_state=42)
model.fit(X)



# 获取聚类中心和数据点的分配
centers = model.cluster_centers_
labels = model.labels_

# 定义颜色列表
colors = ['r', 'g', 'b', 'k','o','y','p']


# 尝试绘制一下代价函数-K曲线

def cost(K):
    model = KMeans(n_clusters=K, random_state=42)
    model.fit(X)
    cost=model.inertia_
    return cost

Cost=np.zeros((19,2))

for k in range(1,20):
    Cost[k-1]=[k,cost(k)]


# 创建一个图窗,左边用来展示K=4的聚类情况,右边用来绘制代价函数曲线

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 绘制数据点
for i in range(K):
    ax1.scatter(X[labels == i][:, 0], X[labels == i][:, 1], c=colors[i], label=f'Cluster {i+1}')

# 绘制聚类中心
ax1.scatter(centers[:, 0], centers[:, 1], c='brown', s=150, marker='X', label='Centers')

ax1.set_title("the result of culstering while K=4")
ax1.legend()

# 绘制折线图
ax2.plot(Cost[:,0],Cost[:,1],color='b')

ax2.set_title("the curve of cost and K")

# 调整布局，使得子图之间没有重叠
plt.tight_layout()

plt.show()



