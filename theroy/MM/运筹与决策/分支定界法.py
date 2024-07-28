import numpy as np

from scipy.optimize import linprog

# 定义最优解数组

best=[((999999999,99999999999),9999999999)]

#定义系数向量和系数矩阵

c=np.array([-5,-4])

A=np.array([[3,2],
            [2,1]])

b=np.array([17,10])

#上下界

bounds=[(0,None),(0,None)]


def f(bounds: list, depth=0):
    global best
    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    # 检查 res 是否有效
    if res.success:
        (x1, x2) = res.x
        value = res.fun
        if x1 == int(x1) and x2 == int(x2):
            best.append(((x1, x2), value))
        else:
            if value > min([best[i][1] for i in range(len(best))]):
                return
            if x1 != int(x1):
                f([(0, int(x1)), bounds[1]], depth+1)
                f([(int(x1)+1, None), bounds[1]], depth+1)
            if x2 != int(x2):
                f([bounds[0], (0, int(x2))], depth+1)
                f([bounds[0], (int(x2)+1, None)], depth+1)

f(bounds)

best_value=min([best[i][1] for i in range(len(best))])

for i in range(len(best)):
    if best[i][1]==best_value:
        optimize=best[i][0]

optimize=[float(i) for i in optimize]

print(f'最优解:{optimize:}')
print(f'最优值:{-best_value:}')

