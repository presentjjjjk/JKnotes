import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义常量
m = 2433
M = 2433
M_prime = 1028.876
I_1 = 1500
I_2 = 8000
I_2_prime = 7001.914
g = 9.8
rho = 1025
S = np.pi / 4
k_1 = 80000
k_2 = 10000
k_3 = 683.4558
k_1_prime = 250000
k_2_prime = 1000
k_3_prime = 654.3383
k_4 = 8890.7
L = 1690
omega = 1.7152
f = 3640

# 定义力
def f(t):
    return np.sin(omega * t)

# 定义微分方程组
def equations(t, y):
    x1, v1, x2, v2, theta1, omega1, theta2, omega2 = y

    f1 = k_1 * (x2 - x1)
    f2 = k_2 * (v2 - v1)

    dx1_dt = v1
    dv1_dt = (f1 + f2 + m * g * (1 - np.cos(theta1)) + x1 * v1**2) / m

    dx2_dt = v2
    dv2_dt = (f(t) * np.cos(omega * t) * np.cos(theta2) - (f1 + f2) * np.cos(theta2 - theta1) +
              M * g * (1 - np.cos(theta2)) - rho * g * S * x2 * np.cos(theta2) -
              k_3 * v2 + x2 * v2**2) / (M + M_prime)

    dtheta1_dt = omega1
    domega1_dt = (k_1_prime * (theta2 - theta1) + k_2_prime * (omega2 - omega1)) / I_1

    dtheta2_dt = omega2
    domega2_dt = (L * np.cos(omega * t) - k_1_prime * (theta2 - theta1) -
                  k_2_prime * (omega2 - omega1) - k_3_prime * omega2 - k_4 * theta2) / (I_2 + I_2_prime)

    return [dx1_dt, dv1_dt, dx2_dt, dv2_dt, dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# 初始条件
y0 = [0, 0, 0, 0, 0, 0, 0, 0]

# 时间区间
t_span = (0, 300)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 数值求解
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# 提取结果
t = solution.t
x1, v1, x2, v2, theta1, omega1, theta2, omega2 = solution.y

# 绘制 x1 和 x2
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x1, label='x1(t)', color='blue')
plt.plot(t, x2, label='x2(t)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement x1 and x2 over Time')
plt.legend()

# 绘制 theta1 和 theta2
plt.subplot(1, 2, 2)
plt.plot(t, theta1, label='theta1(t)', color='green')
plt.plot(t, theta2, label='theta2(t)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Angles theta1 and theta2 over Time')
plt.legend()

plt.tight_layout()
plt.show()
