#这个文件用来计算方程常数,包括总方程常数和独立的方程参数

def independent_constant(T_c,p_c,w,n,T):


    R=8.314462618
    #定义一个方程常数数组:
    a=[]
    
    b=[]

    for i in range(n):
        t_1=0.457235*(R*T_c[i])**2/p_c[i]
        t_2=0.077796*R*T_c[i]/p_c[i]
        t_3=(1+(1-(T/T_c[i])**0.5)*(0.37646+1.54226*w[i]-0.26992*w[i]**2))**2
        
        a.append(t_1*t_3)

        b.append(t_2)

    #返回方程常数数组
    
    return a,b


def total_constant(a,b,x,k,n):
    b_t=sum([x[i]*b[i] for i in range(n)])

    a_t=0

    for i in range(n):
        for j in range(n):
            a_ij=(a[i]*a[j])**0.5*(1-k[i][j])
            a_t =a_t+ x[i]*x[j]*a_ij
    return a_t,b_t






