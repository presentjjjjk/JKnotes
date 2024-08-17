import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f

# 设定F分布的参数
d1 = 500  # 自由度1
d2 = 20  # 自由度2
size = 1000  # 样本数量

# 生成F分布数据
X = f.rvs(d1, d2, size=size)


# 正态化
# 设定不同的参数,看看正态化的效果
b = [10e0,10e1,10e2,10e3,10e4]

Y = [np.log(X) + b[i] for i in range(len(b))]

# 画图,用subplot展示正态化前后的差距
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 原始数据直方图
axes[0, 0].hist(X, bins=30, alpha=0.75, color='blue')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# 正态化后的直方图
for i in range(5):
    ax = axes[(i + 1) // 3, (i + 1) % 3]
    ax.hist(Y[i], bins=30, alpha=0.75, color='blue')
    ax.set_title(f'Normalized Data with b={b[i]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')



# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()
