# Hartree-fork实现

之前在调用计算函数的时候,只调用了HF方法,但是为了加速计算,调用函数的时候可以指明用特殊的HF方法,例如对于闭壳层(自旋成对的体系),可以调用`RHF`,对于自旋不成对的体系(开壳层),调用`UHF`

## 流程

实现HF方法即求解下述方程组:

$$
\begin{cases} F_{\mu\nu}=h_{\mu\nu}+\sum_{\alpha,\beta}D_{\alpha\beta}\langle\alpha\mu||\beta\nu\rangle, &  \\ D_{\alpha\beta}=\sum_i c^*_{\alpha i} c_{\beta i}, &  \\
FC=SCE
\end{cases}
$$

我们采用迭代求解的方式,具体流程为:

1. 先提前准备好积分$h_{\mu\nu},\langle\alpha\mu||\beta\nu\rangle$
2. 选定一组原子轨道基组
3. 初始猜测一个系数矩阵,求解密度矩阵
4. 代入久期方程,求解系数矩阵
5. 比较解与初始猜测的相似程度,若差距较大,将求解得到的矩阵作为猜测矩阵,代回到第三步.

为了防止陷入局部最优或者收敛到鞍点,我们需要取多个初始猜测值,看一看结果是否一致.

`pyscf`中提供了A-O积分的接口可以让我们快速调用积分,所以实现HF方法会便捷很多,求解本征方程可以采用`scipy`库中的`eig`方法



