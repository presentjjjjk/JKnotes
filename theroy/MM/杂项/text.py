import numpy as np
import pandas as pd
from math import radians

# 初始参数的设定
h = 88  # 集热器总高度
d = 7   # 集热器直径
phi = radians(39.4)  # 纬度

# 定日镜坐标数据的读入
file_path = r'D:\桌面\Y20WPner9fa62862794e6dc82731a5561ce1132f\A题\附件.xlsx'
df = pd.read_excel(file_path)
x_values = df.iloc[:, 0].values  # Assuming first column contains X values
y_values = df.iloc[:, 1].values  # Assuming second column contains Y values
X = np.column_stack((x_values, y_values))

# 函数程序

# 计算太阳时角
def w(ST):
    return np.pi / 12 * (ST - 12)

# 计算太阳赤纬角
def delta(D):
    return np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360))

# 计算太阳高度角
def alpha(D, ST, phi, delta_D):
    return np.arcsin(np.cos(delta_D) * np.cos(phi) * np.cos(w(ST)) + np.sin(delta_D) * np.sin(phi))

# 计算太阳方位角
def gamma(D, ST, phi, delta_D, alpha_value):
    hour_angle = w(ST)
    sin_azimuth = np.sin(hour_angle)
    cos_azimuth = (np.sin(delta_D) - np.sin(alpha_value) * np.sin(phi)) / (np.cos(alpha_value) * np.cos(phi))
    cos_azimuth = np.clip(cos_azimuth, -1.0, 1.0)
    azimuth = np.arccos(cos_azimuth)
    azimuth_deg = np.degrees(azimuth)
    if hour_angle > 0:
        azimuth_deg = 360 - azimuth_deg
    return azimuth_deg

# 计算法向辐照强度
def DNI(D, ST, alpha_value):
    a = 0.4237 - 0.00821 * (6 - 3)**2
    b = 0.5055 + 0.00595 * (6.5 - 3)**2
    c = 0.2711 + 0.01858 * (2.5 - 3)**2
    return 1.366 * (a + b * np.exp(-c / np.sin(alpha_value)))

# 计算余弦效率
def eta_cos(alpha_value, gamma_value, X):
    x, y = X
    z = 4
    a = np.cos(alpha_value) * np.sin(gamma_value)
    b = np.cos(alpha_value) * np.cos(gamma_value)
    c = np.sin(alpha_value)
    m, n, t = -x, -y, 84 - z
    l = np.sqrt(m**2 + n**2 + t**2)
    m, n, t = m / l, n / l, t / l
    V0 = np.array([a, b, c])
    V1 = np.array([m, n, t])
    cos_2theta = np.dot(V0, V1) / (np.linalg.norm(V0) * np.linalg.norm(V1))
    return np.sqrt((1 + cos_2theta) / 2)

# 计算大气透射效率
def eta_at(X):
    x, y = X
    z = 4
    d = np.sqrt(x**2 + y**2 + (z - 84)**2)
    return 0.99321 - 0.0001176 * d + 1.97e-8 * d**2

# 计算阴影遮挡效率
def eta_trunc(D, ST, zuobiao, Y, alpha_value, gamma_value):
    x, y = zuobiao
    z = 4
    a = np.cos(alpha_value) * np.sin(gamma_value)
    b = np.cos(alpha_value) * np.cos(gamma_value)
    c = np.sin(alpha_value)
    m, n, t = -x, -y, 84 - z
    l = np.sqrt(m**2 + n**2 + t**2)
    m, n, t = m / l, n / l, t / l

    x_random = np.random.uniform(-3, 3, 1000)
    y_random = np.random.uniform(-3, 3, 1000)
    n_0 = np.array([a + m, b + n, c + t])
    E_A = np.arctan(n_0[2] / np.sqrt(n_0[0]**2 + n_0[1]**2))
    A_A = np.arctan(n_0[0] / n_0[1])
    T_A = np.array([[-np.sin(E_A), -np.sin(A_A) * np.cos(E_A), np.cos(A_A) * np.cos(E_A)],
                    [np.cos(E_A), -np.sin(A_A) * np.sin(E_A), np.cos(A_A) * np.sin(E_A)],
                    [0, np.cos(A_A), np.sin(A_A)]])

    total_true = 0
    for i in range(1000):
        H_0 = np.array([x_random[i], y_random[i], 0])
        H_1 = np.dot(T_A, H_0) + np.array([x, y, z])
        total = 0
        for j in range(len(Y)):
            if abs(Y[j][0]) < abs(x) and abs(Y[j][1]) < abs(y):
                x_b, y_b = Y[j]
                z_b = 4
                m_b, n_b, t_b = -x_b, -y_b, 84 - z_b
                l_b = np.sqrt(m_b**2 + n_b**2 + t_b**2)
                m_b, n_b, t_b = m_b / l_b, n_b / l_b, t_b / l_b
                n_b0 = np.array([a + m_b, b + n_b, c + t_b])
                E_b = np.arctan(n_b0[2] / np.sqrt(n_b0[0]**2 + n_b0[1]**2))
                A_b = np.arctan(n_b0[0] / n_b0[1])
                T_b = np.array([[-np.sin(E_b), -np.sin(A_b) * np.cos(E_b), np.cos(A_b) * np.cos(E_b)],
                                [np.cos(E_b), -np.sin(A_b) * np.sin(E_b), np.cos(A_b) * np.sin(E_b)],
                                [0, np.cos(A_b), np.sin(A_b)]])
                
                H_2 = np.dot(T_b.T, (H_1 - np.array([x_b, y_b, z_b])))
                V_H = np.dot(T_b.T, np.array([a, b, c]))
                x_2 = (V_H[2] * H_2[0] - V_H[0] * H_2[2]) / V_H[2]
                y_2 = (V_H[2] * H_2[1] - V_H[1] * H_2[2]) / V_H[2]

                if -3 <= x_2 <= 3 and -3 <= y_2 <= 3:
                    total += 2
                    break
                else:
                    V_H = np.dot(T_b.T, np.array([m, n, t]))
                    x_2 = (V_H[2] * H_2[0] - V_H[0] * H_2[2]) / V_H[2]
                    y_2 = (V_H[2] * H_2[1] - V_H[1] * H_2[2]) / V_H[2]

                    if -3 <= x_2 <= 3 and -3 <= y_2 <= 3:
                        total += 1
                        break

        total_true += total

    return 1 - total_true / 2000

# 主函数执行
D = 11  # 示例日期
ST = 9.5  # 示例太阳时间
delta_D = delta(D)
alpha_value = alpha(D, ST, phi, delta_D)
gamma_value = gamma(D, ST, phi, delta_D, alpha_value)

for i in range(len(X)):
    t = eta_trunc(D, ST, [X[i][0], X[i][1]], X, alpha_value, gamma_value)
    if t != 1:
        print(f'{t:.4f}')
