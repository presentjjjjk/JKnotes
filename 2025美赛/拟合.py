import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('十年温度数据.xlsx')

# 转换日期为天数
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
reference_date = df.iloc[0, 0]
days = np.array([(date - reference_date).days for date in df.iloc[:, 0]])
temperatures = np.array(df.iloc[:, 6])  # 使用第七列的温度数据

# 定义傅里叶级数拟合函数（考虑年周期性）
def fourier_func(x, a0, a1, b1, a2, b2, a3, b3, a4, b4):
    w = 2 * np.pi / 365.25  # 基频（一年的周期）
    return a0 + \
           a1 * np.cos(w*x) + b1 * np.sin(w*x) + \
           a2 * np.cos(2*w*x) + b2 * np.sin(2*w*x) + \
           a3 * np.cos(3*w*x) + b3 * np.sin(3*w*x) + \
           a4 * np.cos(4*w*x) + b4 * np.sin(4*w*x)

# 进行拟合
p0 = [np.mean(temperatures), 1, 1, 1, 1, 1, 1, 1, 1]  # 初始参数猜测
popt, pcov = curve_fit(fourier_func, days, temperatures, p0=p0)

# 生成平滑曲线
x_smooth = np.linspace(min(days), max(days), 1000)
y_smooth = fourier_func(x_smooth, *popt)

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

# 绘图
plt.figure(figsize=(15, 8))
plt.scatter(days, temperatures, label='Raw Data', alpha=0.7, s=10)
plt.plot(x_smooth, y_smooth, color='red', label='Fitted Curve', linewidth=3)  # 改为暗红色
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.title('Fourier Series Fitting of 10-Year Temperature Data')
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True)
plt.show()

# 打印拟合参数
print("\nFourier Series Fitting Parameters:")
param_names = ['a0(Mean Value)', 
              'a1(Annual Period cos)', 'b1(Annual Period sin)',
              'a2(Semi-annual Period cos)', 'b2(Semi-annual Period sin)',
              'a3(1/3 Year Period cos)', 'b3(1/3 Year Period sin)',
              'a4(1/4 Year Period cos)', 'b4(1/4 Year Period sin)']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")

# 计算拟合优度 R²
y_pred = fourier_func(days, *popt)
residuals = temperatures - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((temperatures - np.mean(temperatures))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f'\nR-squared: {r_squared:.4f}')

# 计算均方误差（MSE）
mse = np.mean(residuals**2)
print(f'Mean Square Error (MSE): {mse:.4f}')

# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(residuals))
print(f'Mean Absolute Error (MAE): {mae:.4f}')