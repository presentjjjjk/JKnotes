import numpy as np

import pandas as pd

import xgboost as xg

from sklearn.metrics import accuracy_score

# 引入数据集

df = pd.read_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\multi_class_dataset.csv')

# 训练集
X=np.array(df.drop(columns='target').values)[0:800]

y=np.array(df['target'].values)[0:800]

# 测试集

X_1=np.array(df.drop(columns='target').values)[800:]
y_1=np.array(df['target'].values)[800:]


# 使用xgboost训练模型

# 通过对参数的设定,可以提高决策树的预测准确程度
model = xg.XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    random_state=42
)

model.fit(X,y)

y_pred = model.predict(X_1)

# 评估准确率

accuracy = accuracy_score(y_pred,y_1)
print(f"模型的准确率: {accuracy}")

