import tensorflow as tf

import numpy as np

import pandas as pd

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense 

from tensorflow.keras.losses import BinaryCrossentropy

# 加载数据
data = pd.read_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\data.csv')

# 分离特征和标签
X = np.array(data[['Feature1', 'Feature2']].values.tolist() ) # 将特征转换为列表
Y = np.array(data['Label'].tolist() ) # 将标签转换为列表


#创建一个神经网络,Dense是层的类型名称
model = Sequential([
    Dense(units=25,activation='sigmoid'),
    Dense(units=15,activation='sigmoid'),
    Dense(units=1,activation='sigmoid')
])

# 这里使用的损失函数是二元交叉熵,我也不知是啥东东
model.compile(loss=BinaryCrossentropy)

# 最小化损失函数,这一步就相当于确定模型的参数
model.fit(X,Y,epochs=100)

# 假设 new_data 是一个包含新样本特征的 NumPy 数组，形状与 X 类似
new_data = np.array([[3.5, 2.5], [6.0, 2.0], [1.5, 1.0]])  # 示例数据

# 使用训练好的模型进行预测
predictions = model.predict(new_data)

# 打印预测结果
print(predictions)



