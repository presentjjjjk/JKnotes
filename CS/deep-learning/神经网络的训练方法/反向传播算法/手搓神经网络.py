import numpy as np

# part 1
# 先定义激活函数和他们的导数:

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def linear(x):
    return x

def linear_derivative(x):
    return 1

# part 2 
# 定义单层隐藏层

class DenseLayer:
    def __init__(self,input_dim,output_dim,activation):
        # 根据输入和输出的维度初始化权重和输入输出:
        '''
        W总共有output_dim行,input_dim列,是一个矩阵
        b是一个列向量,长度为output_dim
        Z = WX + b
        A = activation(Z)
        '''
        self.W = np.random.randn(output_dim,input_dim)/np.sqrt(output_dim) # 为了防止梯度爆炸,初始权重需要除以维度
        self.b = np.random.randn(output_dim).reshape(-1,1)
        self.Z= None
        self.A_in = None
        self.A_out = None

        # 选一个激活函数
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'linear':
            self.activation = linear
            self.activation_derivative = linear_derivative


    # 定义前向传播函数
    def forward(self,A_in):
        # 目标是返回计算值A
        self.A_in = A_in
        self.Z = self.W@A_in + self.b
        self.A_out = self.activation(self.Z)
        return self.A_out
    
    # 定义反向传播函数
    def backward(self,dA_out):
        # 目标是根据dA计算dW,db,dZ
        # 首先计算dZ
        # dZ = dA * partial A/partial Z= dA * activation'(Z) 
        # 注意这里只能逐元素相乘
        dZ = dA_out * self.activation_derivative(self.Z)
        dW = dZ@ self.A_in.T
        db = np.sum(dZ,axis=1,keepdims=True)
        # 为了方便传播,需要返回dA_in
        # dA_in = dZ*partial Z/partial A_in 
        dA_in = self.W.T @ dZ
        return dA_in,dW,db
    
    # 定义更新参数函数
    def update(self,dW,db,learning_rate):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

# 定义神经网络的类
class NN:
    def __init__(self,layer_config):
        self.layers = []
        for (input_dim,output_dim,activation) in layer_config:
            layer = DenseLayer(input_dim,output_dim,activation)
            self.layers.append(layer)
        
    # 定义神经网络中的前向传播,反向传播
    def forward(self,X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self,dA_out):
        dA = dA_out
        grads = []
        for layer in reversed(self.layers):
            dA_in,dW,db = layer.backward(dA)
            # 保留每一层的梯度
            grads.append((dW,db))
            dA = dA_in
        return grads[::-1]
    
    # 定义更新参数函数
    def update(self,grads,learning_rate):
        for layer,grad in zip(self.layers,grads):
            layer.update(grad[0],grad[1],learning_rate)

# part 3
# 定义代价函数以及最后一层的dA
def cost_function(y_pred,y_true):
    return np.mean((y_pred-y_true)**2)

def cost_derivative(y_pred,y_true):
    return 2*(y_pred-y_true)/y_true.shape[0] 

# part 4
def history(model, X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size):
    train_loss = []
    test_loss = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[1])
        batch_loss_history = []
        for i in range(0,X_train.shape[1],batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[:,batch_indices]
            y_batch = y_train[batch_indices]
            # 前向传播
            y_pred_batch = model.forward(X_batch)
            cost = cost_function(y_pred_batch,y_batch)
            batch_loss_history.append(cost)
            # 反向传播
            grads = model.backward(y_batch)
            # 更新参数
            model.update(grads,learning_rate)
        loss_history = batch_loss_history
        train_loss.append(np.mean(loss_history))
        
        # 每个epoch后计算测试集损失
        y_test_pred = model.forward(X_test)
        test_cost = cost_function(y_test_pred, y_test)
        test_loss.append(test_cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss_history[-1]:.4f} | Test Loss: {test_cost:.4f}")

    return train_loss, test_loss

# 生成具有真实关系的测试数据（替换原有随机数据）
def generate_data(samples=1000, input_dim=10):
    # 创建真实权重矩阵（从输入到输出的映射关系）
    true_weights = np.random.randn(1, input_dim) * 2.0
    true_bias = np.random.randn(1)
    
    # 生成特征数据（加入少量噪声）
    X = np.random.randn(input_dim, samples) * 1.5
    # 生成带噪声的标签数据（可学习的关系）
    y = true_weights @ X + true_bias + np.random.normal(0, 0.2, samples)
    
    return X, y.flatten()

# 数据标准化
def standardize_data(X, y):
    # 标准化输入特征
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - X_mean) / X_std
    
    # 标准化输出标签
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, (X_mean, X_std, y_mean, y_std)

# 生成并划分数据集
X, y = generate_data(samples=20000)  # 增加样本量
X, y, scalers = standardize_data(X, y)

# 划分训练集和测试集（80%训练，20%测试）
split_idx = int(0.8 * X.shape[1])
X_train, X_test = X[:, :split_idx], X[:, split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 定义神经网络
model = NN([
    (10, 16, 'relu'),    # 输入层到隐藏层1
    (16, 8, 'relu'),     # 隐藏层1到隐藏层2
    (8, 1, 'linear')    # 输出层
])

# 调整训练参数
history_train, history_test = history(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    learning_rate=0.001,  # 更小的学习率
    epochs=1000,
    batch_size=64
)

from matplotlib import pyplot as plt

# 可视化训练过程（显示训练和测试曲线）
plt.plot(history_train, label='Train Loss')
plt.plot(history_test, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.show()

# 新增预测结果散点图
plt.figure(figsize=(10, 6))
y_pred = model.forward(X_test).flatten()
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Model Predictions vs True Values')
plt.grid(True)
plt.show()
