# 重量数组

weight=[5,4,6,3]

#价值数组

value=[10,40,30,50]

n=4

C=10


def f(weight,value,C,n):
    # 定义一个dp表,来代表所有可能出现的情况,第几行就代表几个物品构成的实例,列就代表背包的容量从0遍历到w
    dp=[[0]*(C+1) for _ in range(n+1)]

    # 定义一个判断表,表示该物品是否被选中
    judge=[[0]*(C+1) for _ in range(n+1)]

    #开始遍历列表内的所有情况
    for i in range(1,n+1):
        for j in range(1,C+1):
            if j>=weight[i-1]:
                if dp[i-1][j]>=dp[i-1][j-weight[i-1]]+value[i-1]:
                    dp[i][j]=dp[i-1][j]
                    judge[i][j]=False
                else:
                    dp[i][j]=dp[i-1][j-weight[i-1]]+value[i-1]
                    judge[i][j]=True

                    
            else:
                dp[i][j]=dp[i-1][j]
                judge[i][j]=False

    return dp[n][C],judge

# 回溯这个判断表
a=[]

j=C

total_value,judge=f(weight,value,C,n)

for i in range (n,0,-1):
    if judge[i][j]:
        a.append(i-1)
        j=j-weight[i-1]

print(total_value,a)

