import numpy as np

from jax import hessian

# 先定义函数

f=lambda x:x[0]**2+2*x[1]**2-2*x[0]*x[1]-2*x[1]

f_1=lambda x: 2*x[0]-2*x[1]

f_2=lambda x: 4*x[1]-2*x[0]-2

#初值
x=np.array([2,2])

while True:

    #定义梯度向量
    grd=np.array([f_1(x),f_2(x)])

    # 定义搜索向量,使用np.linalg.norm来计算模长
    p=grd/np.linalg.norm(grd)

    #定义黑塞矩阵,这里注意hessian的输入参数需要是float

    H=hessian(f)([float(i) for i in x])


    # 定义近似最佳步长:

    t=np.dot(np.transpose(grd),p)/(np.dot(np.dot(np.transpose(p),H),p))

    # 更新x
    x=x-t*p

    # 终止条件

    if np.linalg.norm(grd)<1e-9:
        break

print('极小值点:',x)
print('极小值:',f(x))


