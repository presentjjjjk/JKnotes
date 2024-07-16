import sympy as sp
T_c,V_c=map(float,input('请输入临界参数').split())
#定义符号变量,这里用的是状态参数p,V,T,有两个就行,再加上两个未知参数a,b
V,T,a,b=sp.symbols('V T a b')
R=8.3145
#可以直接用符号变量定义函数表达式
#例如这里的是范德华方程:
p=R*T/(V-b)-a/V**2

y1=sp.diff(p,V)
y2=sp.diff(p,V,2)

y1_v=y1.subs({T:T_c,V:V_c})#接受一个字典作为参数
y2_v=y2.subs({T:T_c,V:V_c})

#得到的是带参的表达式,下一步要解方程
s=sp.solve((y1_v, y2_v), (a, b))
print(s)
#返回的是以一个a和b的元组为元素的列表,得到的值还是一个符号表达式,需要使用sympy中的evalf方法转化成数值
# 计算 a 和 b
a_v=s[0][0].evalf()
b_v=s[0][1].evalf()

print(f'临界参数为a={a_v:.2f},b={b_v:.2f}')
