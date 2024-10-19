# B-O近似

在引入B-O近似之前,我们首先要引入一个原子单位制,来对薛定谔方程进行化简:

## 原子单位制

原始的单电子的薛定谔方程为:

$$
(-\frac{\hbar^2}{2m}\nabla^2-\frac{e}{4\pi \varepsilon_0r})\psi=E\psi
$$

考虑做如下变换:

$$
(x,y,z)=\lambda(x',y',z')
$$

我们把长度单位集中到$\lambda$上,所以薛定谔方程变为:

$$
(-\frac{\hbar^2}{2m\lambda^2}\nabla'^2-\frac{e}{4\pi \varepsilon_0\lambda}\frac{1}{r'})\psi=E\psi
$$

不难发现,系数的量纲全部变为能量,调整$\lambda$的值,令:

$$
\frac{\hbar^2}{m\lambda^2}=\frac{e}{4\pi \varepsilon_0\lambda}
$$

得到:

$$
\lambda=\frac{4\pi\varepsilon_0 \hbar^2}{me}
$$

这个东西就是所谓的波尔半径,薛定谔方程左右同除以系数:

$$
E_a=\frac{\hbar^2}{m\lambda^2}=\frac{e}{4\pi \varepsilon_0\lambda}
$$

就得到化简后的薛定谔方程:

$$
(-\frac{1}{2}\nabla'^2-\frac{1}{r'})\psi=E'\psi
$$

长度用几个几个波尔半径来描述,能量用几个几个$E_a$来描述,这是一种有效的简化手段.

## B-O近似的基本内容

事实上,波恩-奥本海默近似的内容有两条,之前学习的过程中往往只学习了第一条,但是第二条也是非常重要的.

1. 由于$m_e\ll m_N$,所以电子比核运动要快的多,在计算电子的波函数的时候,可以把核视作静止的处理.
2. 在处理核的时候,由于电子运动太快了,所以核相对电子的势能是一个平均化的势能,核只能感受到一个平均势能.

## B-O近似的导出

考虑一个核的总哈密顿算子:

$$
\hat{H}_M=\hat{H}_e+\hat{T}_N=-\frac{1}{2}\sum_{i=1}^n\nabla_{i}^2-\sum_{i<j} \frac{1}{|r_i-r_j|}-\sum_{A,i}\frac{Z_A}{|R_A-r_i|}+\sum_{A<B}\frac{1}{2m_N}\frac{1}{|R_A-R_B|}+\frac{1}{2}\sum_{A}\nabla^2_A
$$

首先,根据B-O近似的假设,核的位置$\{ r_A \}$可以视作静态参数,根据这一系列参数,我们可以求解电子的本征方程:

$$
\hat{H_e}\Psi_i^e=E_i^e\Psi_i^e
$$

求解这个本征方程的任务交给计算方法来做,假设我们已经顺利求解这个方程了,我们可以得到无穷多个本征向量,这组本征向量构成希尔伯特空间的一组完备正交基,所以,考虑分子的本征方程的解可以用这组基线性表示:

$$
|\Psi_M\rangle=\sum_i \chi_i|\Psi_i^e\rangle
$$

不难发现,线性展开的系数$\chi_i$会与原子核的坐标$\{ r_A \}$有关.将上述线性表示代回本征方程中并且左乘一个$\Psi_j^e$:

$$
\langle \Psi_j^e|(\hat{H}_e+\hat{T}_N)\sum_i \chi_i|\Psi_i^e\rangle=E_M\sum_i\chi_i\langle \Psi_j^e|\Psi_i^e\rangle
$$

等式的右边为:

$$
E_M\sum_i\chi_i\langle \Psi_j^e|\Psi_i^e\rangle=E_M\chi_j
$$

等式左边电子的哈密顿算子部分:

$$
\langle \Psi_j^e|\hat{H}_e\sum_i \chi_i|\Psi_i^e\rangle=E_j^e\chi_j
$$

关键在于原子核的哈密顿量部分:

$$
\frac{-1}{2m_A}\langle \Psi_j^e|\sum_A \nabla_A^2\sum_i \chi_i|\Psi_i^e\rangle
$$

上面的这个式子过于复杂,考虑一个核的情况:

$$
\frac{-1}{2m_A}\sum_i\langle \Psi_j^e|\nabla_A^2 \chi_i|\Psi_i^e\rangle=\frac{-1}{2m_A}\sum_i\langle \Psi_j^e|((\nabla_A^2 \chi_i)|\Psi_i^e\rangle+\chi_i\nabla_A^2|\Psi_i^e\rangle +2\nabla_A \chi_i\nabla_A|\Psi_i^e\rangle)
$$

稍微整理一下:

$$
\frac{-1}{2m_A}\sum_i \nabla_A^2\chi_i \langle \Psi^e_j|\Psi_i^e\rangle+\chi_i\langle \Psi^e_j|\nabla_A^2|\Psi_i^e\rangle+2\nabla_A\chi_i\langle \Psi_j^e|\nabla_A|\Psi_i^e\rangle
$$

考虑原子核的轻微移动不会导致电子的波函数发生剧烈变化,于是引入 **绝热近似** ,令非绝热耦合项近似为0:

$$
\begin{cases} \langle \Psi_j^e|\nabla_A|\Psi_i^e\rangle\approx 0, &  \\\langle \Psi^e_j|\nabla_A^2|\Psi_i^e\rangle\approx 0 , &  \end{cases}
$$

所以核部分就化简为:

$$
-\frac{1}{2m_A}\nabla_A^2 \chi_j
$$

综合整理一下式子:

$$
(-\frac{1}{2m_A}\nabla_A^2+E_j^e) \chi_j=E_M \chi_j
$$

把前面这一坨定义成核的哈密顿算子,系数$\chi_j$定义成j号核的波函数,于是我们就得到了B-O近似下的分子薛定谔方程组:

$$
\begin{cases} \hat{H_e}\Psi_i^e=E_i^e\Psi_i^e, &  \\ \hat{H}_N \chi_j=E_M\chi_j, & \\
|\Psi_M\rangle=\sum_i \chi_i|\Psi_i^e\rangle
\end{cases}
$$

## 势能面

固定原子核后,我们可以解出一个电子的基态能量$E^e$,这个基态能量是核的空间排布的函数,这个函数面就是化学中的势能面:

$$
f(r_1,r_2, \ldots )=E^e(\{ r_A \})
$$