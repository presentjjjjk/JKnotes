# 这个文件用来计算逸度系数

from math import log,sqrt,exp

def yidu_c(b,a_t,b_t,p,V,T,t):
    
    R=8.314462618
    
    Z=p*V/R/T

    re=b/b_t*(Z-1)-log(p*(V-b_t)/(R*T))+a_t/(2*sqrt(2)*b_t*R*T)*(b/b_t-2/a_t*t)*log((V+(sqrt(2)+1)*b_t)/(V-(sqrt(2)-1)*b_t))
    
    re0=exp(re)

    return re0