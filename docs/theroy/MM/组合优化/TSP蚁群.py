import random as rd

from math import sqrt
import matplotlib.pyplot as plt

#定义蚂蚁个数

m=10

#给出城市坐标
zuobiao = [
    (34.05, -118.25),
    (40.71, -74.01),
    (37.77, -122.42),
    (51.51, -0.13),
    (48.85, 2.35),
    (35.68, 139.65),
    (55.75, 37.62),
    (39.90, 116.40),
    (-33.87, 151.21),
    (-23.55, -46.63),
    (19.43, -99.13),
    (1.29, 103.85),
    (41.89, 12.49),
    (40.42, -3.70),
    (52.52, 13.40),
    (28.61, 77.21),
    (39.92, 32.85),
    (-26.20, 28.04),
    (35.68, -0.63),
    (-34.61, -58.38)
]


#定义距离函数和总长函数

def distance(i:int,j:int):
    dis=sqrt((zuobiao[i][0]-zuobiao[j][0])**2+(zuobiao[i][1]-zuobiao[j][1])**2)
    return dis

def length(a:list):
    l=0
    for i in range(len(a)-1):
        l=l+distance(a[i],a[i+1])
    l=l+distance(a[-1],a[0])
    return l



#定义迭代次数和最大迭代次数

n=0
max_n=1000

#定义最短路径和最短路径长度

best_length=1000000000000
best_path=[]

#定义信息素浓度字典,初始化信息素，随便一个常数
tau={}

for i in range(20):
    for j in range(20):
        if i!=j:
            tau[(i,j)]=1




#定义蒸发率
r=0.1
#蒸发率

#定义概率参数
a=1
b=2
#b是启发参数，通常要大一点，a和b应该都要大于1，之前两个都设置小于1算法不收敛


while n<max_n:
    #定义所有蚂蚁的路径列表：

    ant_list_sum=[]

    #定义所有蚂蚁的边集的集合
    slide=[]

    for num_of_ant in range(m):
        #定义出发地点和路径列表
        t=rd.randint(0,19)
        path=[]
        path.append(t)


        while(len(path)<20):

            #定义一个转移概率字典，来决定蚂蚁的下一步决策
            p={}

            for j in range(20):
                if j not in path:
                    s=tau[(t,j)]**a*(1/distance(t,j))**b/(sum([tau[(t,k)]**a*(1/distance(t,k))**b for k in range(20) if k not in path]))
                    p[j]=s
            
            
            #用轮盘赌选法选取蚂蚁前往的下一个城市
            p_0=rd.random()

            # 我们把数轴按照概率进行划分
            cumulative_prob = 0.0
            for city, prob in p.items():
                cumulative_prob += prob
                if cumulative_prob >= p_0:
                    t = city
                    break
            
            path.append(t)
        
        #把这只蚂蚁加到路径集合当中去
        ant_list_sum.append(path)
        #定义一个k号蚂蚁的边集
        slide_k=[]
        for i in range(len(path)-1):
            slide_k.append((path[i],path[i+1]))
        slide_k.append((path[-1],path[0]))

        #把它加入到总边集合中去
        slide.append(slide_k)

    
    #所有的蚂蚁都跑了一边，现在更新信息素
    for i in range(20):
        for j in range(20):
            if i!=j:
                s=0
                for x in range(len(ant_list_sum)):
                    if (i,j) in slide[x] or (j,i) in slide[x]:
                        s=s+1/length(ant_list_sum[x])
                tau[(i,j)]=(1-r)*tau[(i,j)]+s
    
    n=n+1
    
    
    if min([length(path) for path in ant_list_sum])<best_length:
        best_length=min([length(path) for path in ant_list_sum])
        
        for path in ant_list_sum:
            if(length(path)==min([length(path) for path in ant_list_sum])):
                best_path=path[:]

print(f'best_length:{best_length:}')
print(f'best_path:{best_path:}')

plt.figure(1)

x=[zuobiao[i][0] for i in range(20)]
y=[zuobiao[i][1] for i in range(20)]

plt.scatter(x,y)

for i in range(19):
    plt.plot(
        [zuobiao[best_path[i]][0], zuobiao[best_path[i+1]][0]],  # X 坐标
        [zuobiao[best_path[i]][1], zuobiao[best_path[i+1]][1]]   # Y 坐标
    )
    
# 最后一条线段，从最后一个城市回到第一个城市
plt.plot(
    [zuobiao[best_path[19]][0], zuobiao[best_path[0]][0]],  # X 坐标
    [zuobiao[best_path[19]][1], zuobiao[best_path[0]][1]]   # Y 坐标
)

plt.show()

                



