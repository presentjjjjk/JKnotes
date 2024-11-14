# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 引入数据集

df = pd.read_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\multi_class_dataset.csv')
X=np.array(df.drop(columns='target').values)
y=np.array(df['target'].values)

# 拆分训练集和测试集:

# 拆分数据集，80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model=RandomForestClassifier(
    n_estimators=200, # 树的数量,代表森林的规模
    max_depth=10, # 最大迭代深度
    min_samples_split=5, # 停止分裂的最小样本数
    random_state=42 # 随机种子,可以确保每次运行构建的森林是相同的
)

model.fit(X_train,y_train)

# 使用测试集来评估模型的效果

y_pred=model.predict(X_test)

# 评估准确率

accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率: {accuracy}")