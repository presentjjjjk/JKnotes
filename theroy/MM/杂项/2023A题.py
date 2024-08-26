import numpy as np
import sympy as sp
import pandas as pd
from math import radians
from scipy.integrate import dblquad

# 初始参数的设定

# 集热器总高度
h=88 

#集热器直径
d=7

#纬度

phi=radians(39.4)

# 定日镜坐标数据的读入
# Load the Excel file
file_path = r'D:\桌面\Y20WPner9fa62862794e6dc82731a5561ce1132f\A题\附件.xlsx'
df = pd.read_excel(file_path)
# Assuming the coordinates are in two columns, e.g., 'X' and 'Y'
# Adjust the column names based on the actual content of your file
x_values = df.iloc[:, 0].values  # Assuming first column contains X values
y_values = df.iloc[:, 1].values  # Assuming second column contains Y values
# Stack the coordinates into a matrix X
X = np.column_stack((x_values, y_values))


# 函数程序

# 计算太阳时角

def w(ST):
    return np.pi/12*(ST-12)

# 计算太阳赤纬角

def delta(D):
    return np.arcsin(np.sin(2*np.pi*D/365)*np.sin(2*np.pi*23.45/360))

# 计算太阳高度角

def alpha(D,ST):
    return np.arcsin(np.cos(delta(D))*np.cos(phi)*np.cos(w(ST))+np.sin(delta(D))*np.sin(phi))

# 计算太阳方位角

def gamma(D,ST):
    return np.arccos((np.sin(delta(D))-np.sin(alpha(D,ST))*np.sin(phi))/(np.cos(alpha(D,ST))*np.cos(phi)))

# 计算法向辐照强度

def DNI(D,ST):
    a=0.4237-0.00821*(6-3)**2
    b=0.5055+0.00595*(6.5-3)**2
    c=0.2711+0.01858*(2.5-3)**2
    return 1.366*(a+b*np.exp(-c/np.sin(alpha(D,ST))))



# 计算各个光学效率

# 余弦效率
def eta_cos(D,ST,X):
    x=X[0]
    y=X[1]
    z=4

    # 求这个点和定日镜法向的夹角
    # 先求定日镜中心和集热器中心的连线
    # 给定太阳高度角,给出入射光锥中心的单位向量

    # 定义光线方向向量
    a=np.cos(alpha(D,ST))*np.sin(gamma(D,ST))
    b=np.cos(alpha(D,ST))*np.cos(gamma(D,ST))
    c=np.sin(alpha(D,ST))

    # 与集热器连线的方向向量

    m=0-x 
    n=0-y 
    t=84-z 

    # 归一化
    l=np.sqrt(m**2+n**2+t**2)
    m=m/l
    n=n/l
    t=t/l

    V0=np.array([a,b,c])
    V1=np.array([m,n,t])

    # 夹角余弦值

    cos_2theta=np.dot(V0,V1)/(np.linalg.norm(V0)*np.linalg.norm(V1))

    theta=np.arccos(np.sqrt((1+cos_2theta)/2))

    return np.cos(theta)

# 大气透射效率

def eta_at(X):
    x=X[0]
    y=X[1]
    z=4

    # 计算到集热塔中心的距离

    d=np.sqrt(x**2+y**2+(z-84)**2)

    return 0.99321-0.0001176*d + 1.97*(1e-8)*d**2
    

# 阴影遮挡效率

# 这里的Y是已经经过吸收塔阴影判别后的坐标矩阵,坐标则是被判别的定日镜
def eta_trunc(D,ST,zuobiao,Y):

    x=zuobiao[0]
    y=zuobiao[1]
    z=4
    # 还是先写入射光线和反射光线
    # 定义光线方向向量
    a=np.cos(alpha(D,ST))*np.sin(gamma(D,ST))
    b=np.cos(alpha(D,ST))*np.cos(gamma(D,ST))
    c=np.sin(alpha(D,ST))

    # 与集热器连线的方向向量

    m=0-x 
    n=0-y 
    t=84-z 

    # 归一化
    l=np.sqrt(m**2+n**2+t**2)
    m=m/l
    n=n/l
    t=t/l

    #蒙特卡洛模拟
    #x,y方向上随机取1000个点
    x_random = np.random.uniform(-3, 3, 1000)
    y_random = np.random.uniform(-3, 3, 1000)

    # 被判别定日镜的法向向量

    n=np.array([a+m,b+n,c+t])

    #求方位角和仰角

    E_A=np.arctan(n[2]/np.sqrt(n[0]**2+n[1]**2))
    A_A=np.arctan(n[0]/n[1])

    # 地面变换矩阵
    T_A=np.array([[-np.sin(E_A),-np.sin(A_A)*np.cos(E_A),np.cos(A_A)*np.cos(E_A)],
                 [np.cos(E_A),-np.sin(A_A)*np.sin(E_A),np.cos(A_A)*np.sin(E_A)],
                 [0,np.cos(A_A),np.sin(A_A)]])
    
    total_true=0
    
    # 开始对这个定日镜执行遮挡判别
    for i in range(1000):
        H_0=np.array([x_random[i],y_random[i],0])
        # 先变换到地面系
        H_1=np.dot(T_A,H_0)+np.array([x,y,z])

        # 定义这个点的遮挡权重

        total=0

        # 遍历其他可能的定日镜
        for j in range(len(Y)):
            if abs(Y[j][0])<abs(x) and abs(Y[j][1])<abs(y):
                x_b=Y[j][0]
                y_b=Y[j][1]
                z_b=4
                # 从地面系变换到B系
                # 要先知道这个定日镜的法向仰角和方位角
                m_b=0-x_b 
                n_b=0-y_b 
                t_b=84-z_b 

                # 归一化
                l=np.sqrt(m_b**2+n_b**2+t_b**2)
                m_b=m_b/l
                n_b=n_b/l
                t_b=t_b/l

                n_b=np.array([a+m_b,b+n_b,c+t_b])

                E_b=np.arctan(n_b[2]/np.sqrt(n_b[0]**2+n_b[1]**2))
                A_b=np.arctan(n_b[0]/n_b[1])

                # B系变换矩阵

                T_b=np.array([[-np.sin(E_b),-np.sin(A_b)*np.cos(E_b),np.cos(A_b)*np.cos(E_b)],
                            [np.cos(E_b),-np.sin(A_b)*np.sin(E_b),np.cos(A_b)*np.sin(E_b)],
                            [0,np.cos(A_b),np.sin(A_b)]])
                
                H_2=np.dot(T_b.T,(H_1-np.array([x_b,y_b,z_b])))

                # 光线也进行变换

                V_H=np.dot(T_b.T,np.array([a,b,c]))

                # 求解坐标

                x_2=(V_H[2]*H_2[0]-V_H[0]*H_2[2])/(V_H[2])
                y_2=(V_H[2]*H_2[1]-V_H[1]*H_2[2])/(V_H[2])

                if -3<=x_2<=3 and -3<=y<=3:
                    total=total+2
                    break
                else:
                    V_H=np.dot(T_b.T,np.array([m,n,t]))
                    x_2=(V_H[2]*H_2[0]-V_H[0]*H_2[2])/(V_H[2])
                    y_2=(V_H[2]*H_2[1]-V_H[1]*H_2[2])/(V_H[2])

                    if -3<=x_2<=3 and -3<=y_2<=3:
                        total=total+1
                        break
                    
        total_true+=total
    
    # 返回阴影遮挡率:

    return 1-total_true/2000
            
# 截断效率

def intt(z,theta,sigma):
    return 1/2/np.pi/sigma**2 *np.exp(-(np.cos(theta)**2+z**2)/sigma**2)

def eta_int(D,ST,zuobiao):
    # 先求所有的标准差
    sigma_sum=2.51
    sigma_s=0.94
    sigma_track=0.63
    sigma_bq=np.sqrt(2*sigma_s)

    # 坐标
    x=zuobiao[0]
    y=zuobiao[1]
    z=4

    d=np.linalg.norm(np.array([x,y,z-84]))

    H=6*abs(eta_cos(D,ST,zuobiao))
    W=6

    sigma_ast=np.sqrt(0.5*(H**2+W**2)/4/d)

    sigma_tot=np.sqrt(d**2*(sigma_sum**2+sigma_bq**2+sigma_ast**2+sigma_track**2))

    result, error = dblquad(intt, 0, np.pi, lambda x: 80, lambda x: 88,args=(sigma_tot,))

    return result


