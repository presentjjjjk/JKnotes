import numpy as np
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
    return np.pi / 12 * (ST - 12)

# 计算太阳赤纬角
def delta(D):
    return np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360))

# 计算太阳高度角
def alpha(D, ST):
    global phi
    return np.arcsin(np.cos(delta(D)) * np.cos(phi) * np.cos(w(ST)) + np.sin(delta(D)) * np.sin(phi))

# 计算太阳方位角
def gamma(D, ST):
    global phi
    solar_declination = delta(D)
    solar_elevation = alpha(D, ST)
    
    cos_azimuth = (np.sin(solar_declination) - np.sin(solar_elevation) * np.sin(phi)) / (np.cos(solar_elevation) * np.cos(phi))
    
    # 限制 cos_azimuth 在 [-1, 1] 之间，以避免无效输入
    cos_azimuth = np.clip(cos_azimuth, -1.0, 1.0)
    
    azimuth = np.arccos(cos_azimuth)
    
    return azimuth

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

    return np.sqrt((1+cos_2theta)/2)

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
    #x,y方向上随机取5000个点
    x_random = np.random.uniform(-3, 3, 1000)
    y_random = np.random.uniform(-3, 3, 1000)

    # 被判别定日镜的法向向量

    n_0=np.array([a+m,b+n,c+t])

    #求方位角和仰角

    E_A=np.arctan(n_0[2]/np.sqrt(n_0[0]**2+n_0[1]**2))
    A_A=np.arctan(n_0[0]/n_0[1])

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
                l_b=np.sqrt(m_b**2+n_b**2+t_b**2)
                m_b=m_b/l_b
                n_b=n_b/l_b
                t_b=t_b/l_b

                n_b0=np.array([a+m_b,b+n_b,c+t_b])

                E_b=np.arctan(n_b0[2]/np.sqrt(n_b0[0]**2+n_b0[1]**2))
                A_b=np.arctan(n_b0[0]/n_b0[1])

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

                if -3<=x_2<=3 and -3<=y_2<=3:
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

def intt(x,y,sigma):
    return 1/2/np.pi/sigma**2 *np.exp(-(x**2+y**2)/(2*sigma**2))

def eta_int(D,ST,zuobiao):
    # 先求所有的标准差
    sigma_sum=2.51*1e-3
    sigma_s=0.94*1e-3
    sigma_track=0.63*1e-3
    sigma_bq=np.sqrt(4*sigma_s**2)

    # 坐标
    x=zuobiao[0]
    y=zuobiao[1]
    z=4

    m=0-x 
    n=0-y 
    t=84-z

    d=np.linalg.norm(np.array([x,y,z-84]))

    H=6*abs(1-eta_cos(D,ST,zuobiao))
    W=6*abs(1-eta_cos(D,ST,zuobiao))

    sigma_ast=np.sqrt(0.5*(H**2+W**2))/4/d

    # 反射光线和法向的夹角

    cos_d=np.sqrt(m**2+n**2)/np.sqrt(m**2+n**2+t**2)

    sigma_tot=np.sqrt(d**2*(sigma_sum**2+sigma_bq**2+sigma_ast**2+sigma_track**2)/cos_d)

    result, _ = dblquad(intt, -3.5, 3.5, lambda x: -4, lambda x: 4,args=(sigma_tot,))

    return result

# 下一步就是开始计算

# 先给定时间数组,都在这里取:

time=[9,10.5,12,13.5,15]

# 日期数组

D_values = [-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]

# 光学效率数组

eta_t=[]

# 余弦效率数组
eta_c=[]

# 阴影遮挡效率
eta_zhe=[]

# 截断效率
eta_jie=[]

# 单位镜面面积平均热功率
P=[]


for D in D_values:
    s_eta_t=0
    s_eta_c=0
    s_eta_zhe=0
    s_eta_jie=0
    s_eta_tou=0
    s_P=0

    for ST in time:
        Y=[] # 用来存放没有被集热塔遮住的定日镜

        # 定义一个总光学效率数组
        eta_total=np.ones((len(X)))*0.92


        for i in range(len(X)):
            eta_c_value = eta_cos(D, ST, [X[i][0], X[i][1]])
            eta_jie_value = eta_int(D, ST, [X[i][0], X[i][1]])
            eta_tou_value = eta_at([X[i][0], X[i][1]])

            s_eta_c += eta_c_value
            s_eta_jie += eta_jie_value

            eta_total[i] = eta_c_value * eta_jie_value * eta_tou_value

            # 先进行集热塔阴影遮挡判别
            y=X[i][1]
            x=X[i][0]

            d=7
            h=8
            l=h/np.tan(alpha(D,ST))
            if y>=np.tan(np.pi-gamma(D,ST))*x or y<=np.tan(np.pi-gamma(D,ST))*x-(d/2*np.sin(gamma(D,ST))+l*np.cos(gamma(D,ST))) or y>=np.tan(np.pi/2-gamma(D,ST))*x+d/(2*np.sin(gamma(D,ST))) or y<=np.tan(np.pi/2-gamma(D,ST))*x-d/(2*np.sin(gamma(D,ST))):
                Y.append([X[i][0],X[i][1],i])
            else:
                eta_total[i]=0

        
        for i in range(len(Y)):
            bbb=eta_trunc(D,ST,[Y[i][0],Y[i][1]],X)
            eta_total[Y[i][2]]=eta_total[Y[i][2]]*bbb
            s_eta_zhe+= bbb
            print(bbb)
        
        s_eta_t += sum(eta_total)

        # 定日镜场单位镜面面积输出热功率:
        E=DNI(D,ST)*sum([eta_total[i] for i in range(len(X))])/len(X)
        s_P += E


    eta_t.append(s_eta_t/len(X)/5)
    eta_c.append(s_eta_c/len(X)/5)
    eta_jie.append(s_eta_jie/len(X)/5)
    eta_zhe.append(s_eta_zhe/len(X)/5)
    P.append(s_P/len(X)/5)


# 日期标签
dates = ["1月21日", "2月21日", "3月21日", "4月21日", "5月21日", 
         "6月21日", "7月21日", "8月21日", "9月21日", "10月21日", 
         "11月21日", "12月21日"]

# 将结果填入表格
data = {
    "日期": dates,
    "平均光学效率": eta_t,
    "平均余弦效率": eta_c,
    "平均阴影遮挡效率": eta_zhe,
    "平均截断效率": eta_jie,
    "单位面积镜面平均输出热功率 (kW/m²)": P
}

# 创建DataFrame
df_result = pd.DataFrame(data)

# 将结果保存为Excel文件
output_file_path = r'D:\桌面\输出结果表格.xlsx'
df_result.to_excel(output_file_path, index=False)

print(f"表格已成功生成并保存为 {output_file_path}")
        















