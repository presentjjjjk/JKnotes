from sklearn.datasets import make_classification
import pandas as pd

# 生成数据集
X, y = make_classification(
    n_samples=1000,    # 样本数量
    n_features=20,     # 特征数量
    n_informative=15,  # 具有信息的特征数量
    n_redundant=5,     # 冗余特征数量
    n_classes=10,       # 类别数量
    n_clusters_per_class=2,  # 每个类别的簇数量
    random_state=42    # 随机种子
)

# 将生成的数据集转换为DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y

# 查看数据集的前几行
print(df.head())

# 保存数据集到CSV文件
df.to_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\multi_class_dataset.csv', index=False)
