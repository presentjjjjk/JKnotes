import numpy as np
from scipy.integrate import dblquad
from math import radians

phi = radians(39.4)

# 计算太阳时角
def w(ST):
    return np.pi / 12 * (ST - 12)

# 计算太阳赤纬角
def delta(D):
    return np.arcsin(np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360))

# 计算太阳高度角
def alpha(D, ST):
    return np.arcsin(np.cos(delta(D)) * np.cos(phi) * np.cos(w(ST)) + np.sin(delta(D)) * np.sin(phi))

# 计算太阳方位角
def gamma(D, ST):
    return np.arccos((np.sin(delta(D)) - np.sin(alpha(D, ST)) * np.sin(phi)) / (np.cos(alpha(D, ST)) * np.cos(phi)))

# 余弦效率
def eta_cos(D, ST, X):
    x = X[0]
    y = X[1]
    z = 4

    # 定义光线方向向量
    a = np.cos(alpha(D, ST)) * np.sin(gamma(D, ST))
    b = np.cos(alpha(D, ST)) * np.cos(gamma(D, ST))
    c = np.sin(alpha(D, ST))

    # 与集热器连线的方向向量
    m = 0 - x 
    n = 0 - y 
    t = 84 - z 

    # 归一化
    l = np.sqrt(m**2 + n**2 + t**2)
    m = m / l
    n = n / l
    t = t / l

    V0 = np.array([a, b, c])
    V1 = np.array([m, n, t])

    # 夹角余弦值
    cos_2theta = np.dot(V0, V1) / (np.linalg.norm(V0) * np.linalg.norm(V1))

    theta = np.arccos(np.sqrt((1 + cos_2theta) / 2))

    return np.cos(theta)

# 截断效率
def intt(x, y, sigma):
    return 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / sigma**2)

def eta_int(D, ST, zuobiao):
    # 先求所有的标准差
    sigma_sum = 2.51e-3
    sigma_s = 0.94e-3
    sigma_track = 0.63e-3
    sigma_bq = np.sqrt(2 * sigma_s)

    # 坐标
    x = zuobiao[0]
    y = zuobiao[1]
    z = 4

    d = np.linalg.norm(np.array([x, y, z - 84]))

    H = 6 * abs(eta_cos(D, ST, zuobiao))
    W = 6

    sigma_ast = np.sqrt(0.5 * (H**2 + W**2) / 4 / d)

    sigma_tot = np.sqrt(d**2 * (sigma_sum**2 + sigma_bq**2 + sigma_ast**2 + sigma_track**2))

    result, error = dblquad(intt, -4, 4, lambda x: -3.5, lambda x: 3.5, args=(sigma_tot,))

    return result

# 调用函数 eta_int 示例
D = 11
ST = 9.5
zuobiao = [100, 11]  # 假设的坐标
result = eta_int(D, ST, zuobiao)
print(f"截断效率的结果是: {result}")
