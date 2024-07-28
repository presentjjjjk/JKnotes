
# 重量数组

weight=[5,4,6,3]

#价值数组

value=[10,40,30,50]

#定义最优解数组

t=[]

def V(k,w):
    if k==0:
        return 0
    else:
        if weight[k-1] <= w:
            if V(k-1,w)>V(k-1,w-weight[k-1])+value[k-1]:
                t.insert(0,0)
               
                return V(k-1,w)
            else:
                t.insert(0,1)
                print(t)
                return V(k-1,w-weight[k-1])+value[k-1]
            

        else:
            t.insert(0,0)
            return V(k-1,w)
        
a=V(4,10)

print(a)
print(t)