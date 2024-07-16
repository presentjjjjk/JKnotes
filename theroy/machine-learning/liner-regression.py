import numpy as np
import matplotlib.pyplot as plt

# 创建一个随机的点列,当做训练集,这里用的是numpy创建的向量,提升运算速度
x = 10 * np.random.rand(20) + 10
y = 10 * np.random.rand(20) + 10

# 给定初始的w和b的值
w, b = 0, 0
a = 0.001  # 学习率
tolerance = 1e-3  # 迭代终止条件

while True:
    # 不断更新梯度下降值
    #在np中,*在向量中的运算是逐元素相乘,类似于点乘
    y_pred = w * x + b
    error = y_pred - y
    decrease_1 = np.sum(error * x) / 100
    decrease_2 = np.sum(error) / 100
    
    w = w - a * decrease_1
    b = b - a * decrease_2
    
    # 迭代终止条件
    if np.abs(decrease_1) < tolerance and np.abs(decrease_2) < tolerance:
        break

print(f'w={w:.2f}, b={b:.2f}')

# 绘制训练集散点图
plt.figure(1)
plt.scatter(x, y)

# 绘制训练得到的直线
x_0 = np.linspace(5, 20, 40)
y_0 = w * x_0 + b
plt.plot(x_0, y_0, label='regression line', color='red')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()

plt.show()
