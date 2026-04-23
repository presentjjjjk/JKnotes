import numpy as np

from jax import hessian,grad

# 定义函数

f=lambda x:x[0]**4+25*x[1]**4+x[0]**2*x[1]**2

# 初值
x=np.array([2,2])

n=0

while True:
    H=hessian(f)([float(i) for i in x])

    x=x-np.dot(np.linalg.inv(H),grad(f)([float(i) for i in x]))

    if np.linalg.norm(grad(f)([float(i) for i in x]))<1e-10:
        break
    
    n=n+1

print('极小值点:',x)
print('极小值:',f(x))
print('迭代次数',n)