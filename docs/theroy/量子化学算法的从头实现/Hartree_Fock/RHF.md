# RHF 

从头手搓一个HF,我们需要用到pyscf包,这个包提供了许多电子积分,只要明确分子对象和基组,就可以直接调用.

电子动能积分:

$$
T_{\mu\nu} = \langle\mu| -\frac{1}{2}\nabla^2 |\nu\rangle = \int \chi_\mu^*(\mathbf{r}) \left(-\frac{1}{2}\nabla^2\right) \chi_\nu(\mathbf{r}) d\mathbf{r}
$$

电子-原子核势能积分:

$$
V_{\mu\nu} = \langle\mu| -\sum_A \frac{Z_A}{r_{1A}} |\nu\rangle = \int \chi_\mu^*(\mathbf{r}_1) \left(-\sum_A \frac{Z_A}{|\mathbf{r}_1 - \mathbf{R}_A|}\right) \chi_\nu(\mathbf{r}_1) d\mathbf{r}_1
$$

重叠积分

$$
S_{\mu\nu} = \langle\mu|\nu\rangle = \int \chi_\mu^*(\mathbf{r}) \chi_\nu(\mathbf{r}) d\mathbf{r}
$$

双电子积分

$$
[\mu\nu|\lambda\sigma]=\langle\mu\lambda|\nu\sigma\rangle = \int\int \chi_\mu^*(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1) \frac{1}{r_{12}} \chi_\lambda^*(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2
$$

## 自旋轨道转空间轨道

软件包采用的积分是空间轨道积分,自旋轨道积分难以直接计算,故要先进行自旋轨道转空间轨道的变换.

### 能量变换

首先,我们约定,圆括号代表空间轨道,并且轨道顺序使用的是化学家记号.

自旋轨道下的能量表达式为:

$$
E=\sum_{i=1}^N\langle i|h|i\rangle+\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \langle ij||ij\rangle
$$

第一项:

$$
\sum_{i=1}^N\langle i|h|i\rangle = \sum_{a=1}^{\frac{N}{2}} \sum_{\sigma \in \{ \alpha, \beta \}} \langle a\sigma|h|a\sigma\rangle = 2 \sum_{a=1}^{\frac{N}{2}} (a|h|a)
$$

电子相互作用项的第一项,库伦排斥项:

$$
\sum_{ij}\langle ij|ij\rangle = \sum_{a=1}^{\frac{N}{2}} \sum_{\sigma \in \{ \alpha, \beta \}} \sum_{b=1}^{\frac{N}{2}} \sum_{\tau \in \{ \alpha, \beta \}} \langle a\sigma b\tau|a\sigma b\tau\rangle = 4 \sum_{a=1}^{\frac{N}{2}} \sum_{b=1}^{\frac{N}{2}} (aa|bb)
$$

即库伦排斥和电子自旋无关

电子相互作用项的第二项,交换互斥项:

$$
\sum_{ij}\langle ij|ji\rangle = \sum_{a=1}^{\frac{N}{2}} \sum_{\sigma \in \{ \alpha, \beta \}} \sum_{b=1}^{\frac{N}{2}} \sum_{\tau \in \{ \alpha, \beta \}} \langle a\sigma b\tau|b\tau a\sigma\rangle = \sum_{a=1}^{\frac{N}{2}} \sum_{b=1}^{\frac{N}{2}} (ab|ba) \int d\omega_1 \chi^*_\sigma(\omega_1)\chi_\tau(\omega_2) \int d\omega_2 \chi^*_\tau(\omega_2)\chi_\sigma(\omega_2) = 2\sum_{a=1}^{\frac{N}{2}} \sum_{b=1}^{\frac{N}{2}} (ab|ba)
$$

最后得到能量的空间轨道积分表示:

$$
E = 2 \sum_{a=1}^{\frac{N}{2}} (a|h|a) + 2 \sum_{a=1}^{\frac{N}{2}} \sum_{b=1}^{\frac{N}{2}} (aa|bb) -  \sum_{a=1}^{\frac{N}{2}} \sum_{b=1}^{\frac{N}{2}} (ab|ba)
$$

### 算子变换

相应的,Fock矩阵的矩阵元也要进行相应的变换:

$$
F_{\mu\nu} = h_{\mu\nu} + J_{\mu\nu} - K_{\mu\nu}
$$

#### 库伦算符

首先,对于hartree算符,由于其是单电子算符,轨道的自旋相同,所以归一化了,没有变化.

对于库伦矩阵,我们有:

$$
J_{\mu\nu} = \sum_{\alpha,\beta}\sum_{i = 1}^N c^*_{\alpha i}c_{\beta i}\iint dx_1dx_2 \phi_\alpha^*(x_1)\phi_\mu^*(x_2)\frac{1}{r_{12}}\phi_\beta(x_1)\phi_\nu(x_2)
$$

i代表分子轨道的下标,值得注意的是,$\beta,\alpha$是对应自旋轨道的原子轨道基函数,所以也有自旋之分,自旋向上的分子轨道只能由自旋向上的原子轨道叠加而成,但是这个原子轨道不一定有电子占据的概念,即一个原子自旋轨道不一定就占据了一个电子,可能分配到这上面的电子个数是分数形式的(极小基除外)

于是,我们将表达式按照自旋去分类展开:

$$
J_{\mu\nu} = \sum_{\alpha = \uparrow,\beta = \uparrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \uparrow}c_{\beta a \uparrow}\langle\alpha\mu|\beta\nu\rangle + \sum_{\alpha = \downarrow,\beta = \downarrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \downarrow}c_{\beta a \downarrow}\langle\alpha\mu|\beta\nu\rangle + \sum_{\alpha = \uparrow,\beta = \downarrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \uparrow}c_{\beta a \downarrow}\langle\alpha\mu|\beta\nu\rangle  + \sum_{\alpha = \downarrow,\beta = \uparrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \downarrow}c_{\beta a \uparrow}\langle\alpha\mu|\beta\nu\rangle 
$$

接着再看积分项,将自旋轨道和空间轨道展开:

$$
\langle\alpha\mu|\beta\nu\rangle = \int dr_1 dr_2 d\omega_1 d\omega_2 \phi_\alpha^*(r_1)\phi_\mu^*(r_2)\frac{1}{r_{12}}\phi_\beta(r_1)\phi_\nu(r_2) \chi^*_\alpha(\omega_1)\chi_\beta(\omega_1)\chi^*_\mu(\omega_2)\chi_\nu(\omega_2)
$$

注意到Fock算符是一个单电子算符,所以原子轨道$\phi_\mu$和$\phi_\nu$本质上是描述同一个电子,所以,这两个原子轨道的自旋部分相同,上式化简为:

$$
(\alpha\beta|\mu\nu)\int d \omega_1 \chi^*_\alpha(\omega_1)\chi_\beta(\omega_1)
$$

当$\alpha,\beta$电子自旋相反时,上述积分为0,否则为1:

$$
J_{\mu\nu} = \sum_{\alpha = \uparrow,\beta = \uparrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \uparrow}c_{\beta a \uparrow}\langle\alpha\mu|\beta\nu\rangle + \sum_{\alpha = \downarrow,\beta = \downarrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \downarrow}c_{\beta a \downarrow}\langle\alpha\mu|\beta\nu\rangle 
$$

对于闭壳层的一个空间分子轨道,原子自旋轨道对分子自旋轨道的贡献是相同的,所以这两项实际上是一样的,我们将下标更换为$\lambda ,\sigma$,这是对空间轨道计数的下标,于是库伦矩阵更改为:

$$
J_{\mu\nu} = 2 \sum_{\lambda,\sigma} D_{\lambda\sigma}(\lambda\sigma|\mu\nu)
$$

此时我们新定义的密度矩阵为:

$$
D_{\lambda\sigma} = \sum_{a = 1}^{\frac{N}{2}} c^*_{\lambda a}c_{\sigma a}
$$


#### 交换算符

$$
K_{\mu\nu} = \sum_{\alpha,\beta}P_{\alpha\beta}\int dr_1dr_2 d\omega_1\omega_2\phi_\alpha^*(r_1)\phi_\mu^*(r_2)\frac{1}{r_{12}}\phi_\nu(r_1)\phi_\beta(r_2) \chi^*_\alpha(\omega_1)\chi_\nu(\omega_1)\chi^*_\mu(\omega_2)\chi_\beta(\omega_2)
$$

我们现在要将$\alpha,\beta$改成对空间轨道的求和变量,首先按照自旋进行展开:

$$
K_{\mu\nu} = \sum_{\alpha = \uparrow,\beta = \uparrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \uparrow}c_{\beta a \uparrow}\langle\alpha\mu|\nu\beta\rangle + \sum_{\alpha = \downarrow,\beta = \downarrow}\sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \downarrow}c_{\beta a \downarrow}\langle\alpha\mu|\nu\beta\rangle + \sum_{\alpha = \uparrow,\beta = \downarrow} \sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \uparrow}c_{\beta a \downarrow}\langle\alpha\mu|\nu\beta\rangle + \sum_{\alpha = \downarrow,\beta = \uparrow} \sum_{a = 1}^{\frac{N}{2}} c^*_{\alpha a \downarrow}c_{\beta a \uparrow}\langle\alpha\mu|\nu\beta\rangle
$$

由于$\mu,\nu$轨道自旋相同,如果$\alpha,\beta$自旋不同,则必定有一项积分为0,所以自旋相反的项全部为0.

对于自旋相同的项,由于不确定$\mu,\nu$所代表的电子自旋,故有且只有一项不为0.将求和指标替换为$\lambda,\sigma$,我们得到新的交换算符的表达式:

$$
K_{\mu\nu} = \sum_{\lambda,\sigma}D_{\lambda\sigma}(\lambda\nu|\mu\sigma)
$$

最后,使用我们新定义的这个密度矩阵,对能量的表达式进行展开,可以得到:

$$
E = 2 \sum_{\lambda,\sigma} D_{\lambda \sigma} h_{\lambda\sigma} + \sum_{\lambda,\sigma,k,l}D_{\lambda k}D_{\sigma l}[2(\lambda k|\sigma l)-(\lambda l|\sigma k)]
$$

这个表达式可以进一步化简,只使用Fock矩阵,密度矩阵和hartree矩阵就能表述,首先,改变后面那一项的求和指标的名字,将$k$和$\sigma$进行调换:

$$
E = 2 \sum_{\lambda,\sigma} D_{\lambda \sigma} h_{\lambda\sigma} + \sum_{\lambda,\sigma,k,l}D_{\lambda \sigma}D_{k l}[2(\lambda \sigma|k l)-(\lambda l| k \sigma)] 
$$

由于积分变量可以随便更改,所以存在交换律:

$$
(ab|cd) = (cd | ba)
$$

所以上式变为:

$$
E = 2 \sum_{\lambda,\sigma} D_{\lambda \sigma} h_{\lambda\sigma} + \sum_{\lambda,\sigma,k,l}D_{\lambda \sigma}D_{k l}[2(k l|\lambda \sigma)-( k \sigma|\lambda l)] = \sum_{\lambda ,\sigma} D_{\lambda \sigma} (2h_{\lambda\sigma} + J_{\lambda \sigma}-K_{\lambda \sigma}) = \sum_{\lambda,\sigma}D_{\lambda \sigma}(h_{\lambda \sigma}+ F_{\lambda \sigma})
$$

整理一下就得到:

$$
E = \mathbf{Tr(D(h+F)^T)}
$$

或者写成时间复杂度更低的运算方式:

$$
E = \sum D \odot (h + F)
$$

## python实现

pyscf提供了轨道积分,通过以下接口调用:

```python
kin = mol.intor('int1e_kin') 动能积分
vnuc = mol.intor('int1e_nuc') 核势能积分
overlap = mol.intor('int1e_ovlp') 重叠积分
eri = mol.intor('int2e') 双电子积分
```

仅仅使用numpy和pyscf,就可以很方便的复现Hartree-Fock算法:

```python
from pyscf import gto,scf
from scipy.linalg import eigh

import numpy as np

mol = gto.M(
    '''
    H 0 0 0; 
    H 0 0 1;
    ''',
    basis='ccpvtz',
    charge=0,
    spin=0,
)

T = mol.intor('int1e_kin')
V = mol.intor('int1e_nuc')
h = T + V

# 重叠积分
S = mol.intor('int1e_ovlp')

# 双电子积分张量
eri = mol.intor('int2e') # [alpha, beta, mu, nu]
iter = 0

_ , C = eigh(h, S)
nocc = mol.nelectron // 2
D = np.einsum('pi,qi->pq', C[:, :nocc], C[:, :nocc])  # 只用占据轨道

# 混合因子
alpha = 0.5
while True:    

    # 构建Fock矩阵
    F = h + 2 * np.einsum('ab,pqab->pq', D, eri) - np.einsum('ab,pbaq->pq', D, eri)

    # 求解久期方程FC = SCE 广义本征值问题
    E, C_prime = eigh(F , S)
    iter += 1
    
    # 计算能量
    E_ele = np.einsum('pq,pq->', D, h + F) 
    print('iter: {}, E_ele: {}'.format(iter, E_ele))

    D_prime = np.einsum('pi,qi->pq', C_prime[:, :nocc], C_prime[:, :nocc])
    if np.allclose(D, D_prime):
        break

    D = (1 - alpha) * D + alpha * D_prime

mf = scf.RHF(mol)
mf.run()
print('HF energy: ', mf.e_tot-mf.energy_nuc())
```

Results:

```
iter: 25, E_ele: -1.6312395304985876
converged SCF energy = -1.1020623237473
HF energy:  -1.631239534667296
```

和软件算的几乎没有区别,这证明我们实现的算法相当可靠




