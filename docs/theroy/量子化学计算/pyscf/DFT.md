# DFT

得益于底层逻辑的设计，DFT的运行速度会比同等精度的自洽场方法要快的多，所以在很多计算场合往往成为工程师们计算的首选，但是，DFT的精度难以提升，所以在一些高精度计算的场合，DFT往往不被采用。

## 从H-F开始

pyscf的文档文件里面给出了一个简单的DFT计算示例,还是从一个简单的氟化氢分子开始.

```python
from pyscf import gto, dft
mol_hf = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz', symmetry = True)
mf_hf = dft.RKS(mol_hf)
mf_hf.xc = 'lda,vwn' # default
mf_hf = mf_hf.newton() # second-order algortihm
mf_hf.kernel()
```

可以看到,构建分子对象的部分依旧保持不变,只不过,构建方法对象的时候,使用的函数由原来的`RHF`变成了`RKS`,同理,这也是针对闭壳层的.

比之自洽场方法不同的是,我们需要制定交换关联泛函,这分为两部分,即分别指定交换泛函和关联泛函.

```python
mf_hf.xc = 'lda,vwn'
```

`lda`是均匀电子气导出的近似交换泛函,`vwn`则是由均匀电子气导出的近似电子关联泛函.

`mf_hf.newton()`是将迭代方法设为了牛顿法,和之前提到的`fast_newton`不同,他只会设置计算方法,不会默认执行运算.

## 交换关联泛函

泛函的选取是需要根据实际需求决定的,更多时候都是经验性的.

pyscf的示例里面列出了大量的常用泛函:

```
#mf.xc = 'svwn' # shorthand for slater,vwn
#mf.xc = 'bp86' # shorthand for b88,p86
#mf.xc = 'blyp' # shorthand for b88,lyp
#mf.xc = 'pbe' # shorthand for pbe,pbe
#mf.xc = 'lda,vwn_rpa'
#mf.xc = 'b97,pw91'
#mf.xc = 'pbe0'
#mf.xc = 'b3p86'
#mf.xc = 'wb97x'
#mf.xc = '' or mf.xc = None # Hartree term only, without exchange
mf.xc = 'b3lyp'
```

目前普适性最好的,最常用的交换关联泛函是`b3lyp`,可以适用于大量的主族元素.

泛函的提出往往是带有目的性的,但是其效果却没有提出者想象的那么好,实际使用的时候需要参考期刊`Journal of Chemical Theory and Computation`和`Journal of Physical Chemistry A`.

泛函的粗略选取可以根据卢天老师的文章进行:

[http://sobereva.com/272]

## 与HF的比较

DFT的计算成本只会比HF略高,但是由于充分考虑了电子关联,其精度通常远高于HF(泛函选取要准确),下面来验证这一点:

```
converged SCF energy = -99.7809511810294
DFT用时: 1.7983689308166504
converged SCF energy = -100.019411269174
scf用时 0.11860132217407227
```

以氟化氢分子为例,DFT计算出的总能量更加符合高精度计算的结果,但是时间成本变高了.

以计算C原子的电子亲和能为例,我们使用HF这种比较粗糙的办法,计算出来的电子亲和能严重偏离实验值,这是因为我们没有考虑电子关联(我已经添加了弥散函数).

现在我们考虑使用DFT方法(泛函为`b3lyp`),看看电子亲和能和实际值差距为多少.得到的结果为`电子亲和能为: 1.3807650016423292 eV`,与文献值1.26eV相比偏大,应该是交换关联泛函还不够精确的原因(我已经对负电荷基组添加了弥散函数,基组为ccpv5z),使用sob老师博文中推荐的`wB97X`基组,计算出来的结果为:`电子亲和能为: 1.2407795028853206 eV`,理论与实验符合比较好,这在一定程度上体现了DFT计算的优越性.

和HF相同的是,DFT的计算方法也有适用于闭壳层体系的`RKS`,适用于开壳层体系的`UKS`,以及更贵的广义方法`GKS`




