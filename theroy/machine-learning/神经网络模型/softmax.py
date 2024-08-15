import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense 

model = Sequential([
    Dense(units=128,activation='relu'),
    Dense(units=64,activation='relu'),
    Dense(units=32,activation='relu'),
    Dense(units=16,activation='relu'),
    Dense(units=10,activation='linear')
])

# 引入分类交叉熵

from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

# 引入数据集

df = pd.read_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\multi_class_dataset.csv')

# 训练集
X=np.array(df.drop(columns='target').values)[0:800]

y=np.array(df['target'].values)[0:800]

# 测试集

X_1=np.array(df.drop(columns='target').values)[800:]
y_1=np.array(df['target'].values)[800:]




model.fit(X,y,epochs=100)

logits = model(X_1) # 注意这里神经网络输出的只是z值,我们还要自己加一个softmax

output=np.array(tf.nn.softmax(logits)) # 可以根据这个得到测试集的输出.

# 下一步,选出测试集的每个输出向量中概率最大的那个,然后将它认为是输出

y_final=np.zeros((200))

for i in range(200):
    y_final[i]=np.argmax(output[i])

# 评估误差
s=0

for i in range(200):
    if y_final[i]!=y_1[i]:
        s=s+1

print(f'准确率为:{1-s/200}')




