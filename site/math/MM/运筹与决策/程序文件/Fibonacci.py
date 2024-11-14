# 先定义斐波那契数列
def Fib(n):
    if n==0 or n==1:
        return 1
    else:
        a,b=1,1
        for i in range((n-1)//2):
            a=a+b
            b=a+b
        if (n-1)%2==0:
            return b
        else:
            return a+b



#定义最大迭代次数

N=1000

#定义迭代次数
n=0

#定义一个很小的数
e=0.001

#定义允许容差
tolerance=1e-9

#给定初始搜索区间
a,b=-999,999

#定义匿名函数

f=lambda x:(x-1)**2

while(n <N-1):
    if n==N-1:
        r=1-Fib(N-n)/Fib(N-n+1)-e
    else:
        r=1-Fib(N-n)/Fib(N-n+1)
    t_1=a+r*(b-a)
    t_2=b-r*(b-a)
    if (f(t_1)<f(t_2)):
        b=t_2
    else:
        a=t_1
    
    if(b-a<tolerance):
        break

    n+=1


print('最优解x=',a)
print('最优值f(x)=',f(a))
print('迭代次数:',n)


