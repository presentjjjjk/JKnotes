
# 重量数组

weight=[5,4,6,3]

#价值数组

value=[10,40,30,50]

#定义最优解数组

t=[0 for _ in range(4)]

def V(k,w):
    if k==0:
        return 0
    else:
        if weight[k-1] <= w:
            if V(k-1,w)>V(k-1,w-weight[k-1])+value[k-1]:
                #t.insert(0,0)
                #在比较大小的时候就会重复调用这个t的添加操作,所以不能这么做
                #解决办法是定义一个数组,每次的重复调用函数会得到相同的操作,这样就行了.
                t[k-1]=0
                return V(k-1,w)
            else:
                t[k-1]=1
        
                return V(k-1,w-weight[k-1])+value[k-1]
            

        else:
            t[k-1]=0
            return V(k-1,w)
        
a=V(4,10)

print(a)
print(t)

