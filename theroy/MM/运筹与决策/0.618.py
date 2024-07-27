# 我们先给定一个很大的区间:

a,b=-999,999

n=0

# 定义一个匿名函数

f= lambda x: (x-1)**2

#给定一个允许容差
e=1e-9

while True:
    t_1=a+0.3819*(b-a)
    t_2=b-0.3819*(b-a)
    if f(t_1)<f(t_2):
        b=t_2
    else:
        a=t_1
    if (b-a)<e:
        print('最小值点:',a)
        print('最小值:',f(a))
        print('迭代次数:',n)
        break
    n=n+1
    