import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# 引入数据集
df = pd.read_csv(r'E:\project\my-project\docs\theroy\machine-learning\神经网络模型\multi_class_dataset.csv')

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target').values, df['target'].values, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建模型
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),  # Dropout层可以防止过拟合
    Dense(units=64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(units=10, activation='softmax')  # 输出层使用softmax
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# 训练模型，使用验证集并加入早停策略
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集准确率为: {accuracy:.2f}')
