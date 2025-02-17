from scipy.optimize import linprog
import numpy as np

c=np.array([5,5.1,5.4,5.5,0.2,0.2,0.2])

A_eq=np.array([[1,0,0,0,-1,0,0],
               [0,1,0,0,1,-1,0],
               [0,0,1,0,0,1,-1],
               [0,0,0,1,0,0,1]])

b_eq=np.array([15,25,35,25])

#上下界约束,如果某一个没有就写None或者np.inf

x_bounds = [(0, 30), (0, 40), (0, 45), (0, 20), (0, None), (0, np.inf), (0, np.inf)]

res=linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

if res.success:
    print('最优值',res.fun)
    print('最优解',res.x)
else:
    print('求解失败')