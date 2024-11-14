import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gurobipy as gp
from gurobipy import GRB

X = np.array([
    [0,0,0],
    [80, 1, 5],
    [69, 48, 4],
    [47, 88, 2],
    [5, 88, 4],
    [86, 27, 1],
    [88, 68, 10],
    [20, 90, 6],
    [52, 13, 4],
    [30, 14, 3],
    [53, 63, 10]
])

# 节点个数
m=len(X)


# 初始参数设定
d=3

T=3000

v=0.1

# 定义问题

model = gp.Model("mip_model")

a=model.addVars(m,m,vtype=GRB.BINARY,name='a')

for i in range(m):
    model.addConstr(a[i,i]==0)

b=model.addVars(m,vtype=GRB.BINARY,name='b')

# 定义转折点坐标(连续变量)

x=model.addVars(m,vtype=GRB.CONTINUOUS,name='x')
y=model.addVars(m,vtype=GRB.CONTINUOUS,name='y')

# 先加原点的约束,原点必须在路径中
model.addConstr(b[0]==1)

# 单点输送

for i in range(m):
    model.addConstr(gp.quicksum(a[i,j] for j in range(m))==b[i])

for j in range(m):
    model.addConstr(gp.quicksum(a[i,j] for i in range(m))==b[j])

# 节点顺序变量 u[i]
u = model.addVars(m, vtype=GRB.CONTINUOUS, name='u')

# 起始点的次序约束（可选）
model.addConstr(u[0] == 0)

# MTZ次环消除约束
for i in range(1, m):
    for j in range(1, m):
        if i != j:
            model.addConstr(u[i] - u[j] + (m - 1) * a[i, j] <= m - 2)

# 确保 u[i] 的范围
for i in range(1, m):
    model.addConstr(u[i] >= 1)
    model.addConstr(u[i] <= m - 1)


# x,y的约束:

model.addConstr(x[0]==0)
model.addConstr(y[0]==0)

for i in range(1,m):
    model.addConstr((x[i]-X[i][0])**2+(y[i]-X[i][1])**2<=d**2)

# 时间约束

# 先增加一个变量用了计算平方根,Gurobi不支持直接计算平方根

dis=model.addVars(m,m,vtype=GRB.CONTINUOUS,name='dis')

for i in range(m):
    for j in range(m):
        model.addConstr(dis[i,j]**2==(x[i]-x[j])**2+(y[i]-y[j])**2)

model.addConstr(gp.quicksum(a[i,j]*dis[i,j] for i in range(m) for j in range(m))<=T*v)


# 设置目标函数
model.setObjective(gp.quicksum(X[i][2]*b[i] for i in range(m)), GRB.MAXIMIZE)


model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")

    # 获取变量
    b_values = model.getAttr("X", b)
    x_values = model.getAttr("X", x)
    y_values = model.getAttr("X", y)
    a_values = model.getAttr('X', a)
    for i in range(1,m):
        print(f'x={x_values[i]:.2f},y={y_values[i]:.2f},是否探索:{b_values[i]}')
    
        # 创建图形和轴
    fig, ax = plt.subplots()
    
    # 画离散点
    ax.scatter(X[:, 0], X[:, 1], color='blue', marker='o')
    
    # 画连接线和圆
    for i in range(m):
        for j in range(m):
            if a_values[i, j] == 1:
                # 连接线
                x_conner = [x_values[i], x_values[j]]
                y_conner = [y_values[i], y_values[j]]
                ax.plot(x_conner, y_conner, color='green')

                # 圆
                circle_1 = patches.Circle((X[i][0], X[i][1]), d, edgecolor='blue', facecolor='none')
                circle_2 = patches.Circle((X[j][0], X[j][1]), d, edgecolor='blue', facecolor='none')
                ax.add_patch(circle_1)
                ax.add_patch(circle_2)



    # 添加标题和标签
    plt.title("path of UAV")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 显示图像
    plt.grid(True)
    plt.show()
    # 设置长宽比例一致
    plt.axis('equal')



else:
    print("No optimal solution found.时间太短,请延长探索时间")










