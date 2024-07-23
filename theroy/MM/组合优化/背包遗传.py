
import random as rd

#先定义种群规模
N=100

#定义进化代数
n=0

zhongqun=[]

best_value=0
best=[]

weight=[4,3,1,5,2,1,3,2,4,3]
value_0=[300,200,150,500,200,100,250,300,350,400]

for i in range(N):
    t=[]
    for i in range(10):
        t.append(rd.randint(0,1))
    zhongqun.append(t)

while True:
    #评价适应度
    #定义一个适应度数组
    value=[i for i in range(N)]

    for i in range(N):
        u=sum([weight[j]*zhongqun[i][j] for j in range(10)])
        v=sum([value_0[j]*zhongqun[i][j] for j in range(10)])
        

        #淘汰掉生成的不合理个体
        if(u>10):
            value[i]=0
        else:
            value[i]=v
    

    #选择适应度最高的进行存储：
    best_value=max(value)
    best=zhongqun[value.index(best_value)]

    #设定概率：
    p=[value[i]/sum(value) for i in range(N)]

    #定义交配空间J
    J=[]


    i=0
    while True:
        if rd.random()<p[i]:
            J.append(zhongqun[i])
        if len(J)==N:
            break
        i += 1
        if i==N:
            i=0
    
    #设定交配概率
    p_0=0.85
    #定义一个交配数组J_0
    J_0=[]
    yubeizhongqun=[]

    for i in range(N):
        if rd.random()<p_0:
            J_0.append(J[i])
        else:
            yubeizhongqun.append(J[i])
    
    #然后再配对
    if len(J_0)==1 or len(J_0)==0:
        yubeizhongqun=J
    else:
        
        #单点配对交配
        i=0
        while(i<len(J_0)-1):
            k=rd.randint(0,9)
            J_0[i],J_0[i+1]=J_0[i][0:k]+J_0[i+1][k:],J_0[i+1][0:k]+J_0[i][k:]
            i=i+2
        
        yubeizhongqun +=J_0

        #变异，定义变异概率

        p_b=0.1

        for ele in yubeizhongqun:
            if rd.random()<p_b:
                k=rd.randint(0,9)
                ele[k]=1-ele[k]
        
        zhongqun=yubeizhongqun
        n=n+1

        if n>=100:
            break

print(f'bestvalue:{best_value:}')
print(f'best:{best:}')
        
            
        



    
   


        