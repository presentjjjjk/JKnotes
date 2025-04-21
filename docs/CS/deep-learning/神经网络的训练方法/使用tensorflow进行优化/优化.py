import tensorflow as tf
import numpy as np

# 定义变量
w = tf.Variable(0,dtype=tf.float32)

cost = tf.add(tf.add(w**2,tf.multiply(-10,w)),25) 
# cost = (w-5)**2

# coefficient = np.array([[1.],[-10.],[25.]])
# x= tf.placeholder(tf.float32,[3,1])
# cost = x[0][0]*w**2-10*w+25

# 定义优化求解器
# 这个写法在tensorflow2.x中被重构了,优化求解器要从keras中导入
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def train_step():
    with tf.GradientTape() as tape:
        cost = (w - 5)**2
    gradients = tape.gradient(cost, [w])
    optimizer.apply_gradients(zip(gradients, [w]))
    return gradients

for epoch in range(1000):
    # 执行单个训练步骤
    grad = train_step()
    
    # 打印当前参数值
    current_w = w.numpy()
    print(f"Epoch {epoch+1}: w = {current_w:.2f}")
    if epoch+1 == 300 or np.linalg.norm(grad)<1e-6:
        break
    

