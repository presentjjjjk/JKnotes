# post-HF方法

## MP2

### 简单的示例

MP是一种post-hartree-fork方法,其主要就是对HF做一个微扰修正,所以执行代码的时候要先run HF才能运行MP,一个简单的MP2的计算代码如下所示:

```python
from pyscf import gto,scf,mp

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

mf.MP2().run()

# MP是在HF的基础上才能运行,所以要先运行HF才能运行MP

# 最后一行可替换为:
# mp.MP2(mf).kernel() or mp.MP2(mf).run()

```

pyscf貌似只支持MP2,没有更高阶的MP微扰方法,pyscf的MP2支持开壳层,闭壳层和不受限计算,其不需要再MP2的方法上额外制定,只需要指定HF计算使用的是哪一种即可.

### 冻结轨道

在活性空间方法中,我们知道可以冻结一些在化学过程中基本上不参与的轨道,比如较为内层的轨道中没有化学家们所说的价电子,同理这些轨道在MP2,CC等计算方法中也可以被冻结,这样我们要考虑的组态就大大减少,可以有效降低CI矩阵的维度,从而降低计算量.同理,一些能级非常之高的轨道也可以被冻结,其在化学反应中也基本不参与.

在pyscf的MP2方法中,冻结轨道是非常容易的,只需要指定`frozen`参数即可.例如在`MP2`的方法中使用`mymp = mp.MP2(mf, frozen=2).run()`,这默认冻结能量最低的两个 **空间轨道**,也就是四个自旋轨道.

有的时候需要更加精细化的控制可以输入一个数组代表被冻结轨道的序号:

```python
# freeze 2 core orbitals
mymp = mp.MP2(mf, frozen=[0,1]).run()
# freeze 2 core orbitals and 3 unoccupied orbitals
mymp = mp.MP2(mf, frozen=[0,1,16,17,18]).run()
```

或者也可以自动冻结核心轨道:`mymp = mp.MP2(mf).set_frozen().run()`,对于元素的核心轨道的设定和ORCA中一致.

## CI

### CISD

和MP2类似,也是要先运行HF才能运行CI方法,以CISD为例:

```python
from pyscf import scf,gto,ci

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
mycc = mf.CISD().run() # 补上电子关联
print('RCISD correlation energy', mycc.e_corr)

mf = scf.UHF(mol).run()
mycc = mf.CISD().run()
print('UCISD correlation energy', mycc.e_corr)

# 补上电子关联也可以用ci模块中的方法来执行,等价代码为:
# myci = ci.CISD(mf).run()
```

运行结果为:

```
converged SCF energy = -99.9873974403489
E(RCISD) = -100.196182976057  E_corr = -0.2087855357081034
RCISD correlation energy -0.20878553570810343
converged SCF energy = -99.9873974403479  <S^2> = 8.0957463e-12  2S+1 = 1
E(UCISD) = -100.1961829759279  E_corr = -0.2087855355799747
UCISD correlation energy -0.2087855355799747
```

### FCI

非常遗憾的是,pyscf不支持CISDT和CISDTQ,但是它支持Full-CI计算:

==请注意,进行FCI计算的时候不要使用太大的基组,如果算力充沛请忽略==

只需要使用fci模块中的FCI方法即可,用法和之前是一样的:

```python
from pyscf import gto,scf,fci

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31g',
    symmetry = True,
)
myhf = scf.RHF(mol).run()

cisolver =fci.FCI(myhf).kernel()

print(cisolver)

# print('E(FCI) = %.12f' % cisolver.kernel()[0])

```

注意,在这里对`cisolver`这个对象是用`run`是不会返回任何东西的,只有用`kernel`方法才会返回结果,我们来观察一下返回的东西:

```
(np.float64(-100.10211465664833), FCIvector([[-9.77930457e-01,  0.00000000e+00,  0.00000000e+00, ...,
             0.00000000e+00, -4.40002184e-08,  2.06191429e-07],
           [ 0.00000000e+00,  1.03109224e-02,  0.00000000e+00, ...,
             6.90889107e-07,  0.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.03109224e-02, ...,
             0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           ...,
           [ 0.00000000e+00,  6.90889107e-07,  0.00000000e+00, ...,
            -1.49704252e-07,  0.00000000e+00,  0.00000000e+00],
           [-4.40002184e-08,  0.00000000e+00,  0.00000000e+00, ...,
             0.00000000e+00,  1.18201743e-08,  9.31735710e-09],
           [ 2.06191429e-07,  0.00000000e+00,  0.00000000e+00, ...,
             0.00000000e+00,  9.31735710e-09,  5.37069768e-09]]))
```

返回了一个元组,里面有两个元素,第一个是full-CI的总能量,第二个叫做FCIvector,应该返回的是不同本征值下对应的系数向量,组合起来就是一个矩阵,非基态能量对应的系数就是整个分子激发态的描述,虽然我们的full-ci基态有激发态分子的贡献,本质上还是描述的是基态分子.

## CC

CC可以提供比CI同等计算资源下更加精确的结果,其用法和前面一样,都需要先执行HF后再执行cc,CCSD要都从cc这个模块中调用.不能直接执行CCSD(T),而是在CCSD计算完毕后,再对`mycc`这个对象执行`mycc.ccsd_t()`方法计算出其和CCSD(T)的能量差,从而得到CCSD(T)的能量.

```python
from pyscf import gto,scf,cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31g',
    symmetry = True,
)
myhf = scf.RHF(mol).run()

mycc =cc.CCSD(myhf).run()

e_ccsd_t=mycc.ccsd_t()
print('e_ccsd_t error',e_ccsd_t)
```
==非常值得注意的一点是,这里的mycc对象要用`run`方法执行,而不是`kernel`方法,因为`kernel`方法会返回你计算的结果,而`run`方法会返回一个对象的地址,从而可以继续调用ccsd_t()这个方法继续执行计算==

返回结果为:

```
converged SCF energy = -99.9593211672815
<class 'pyscf.cc.ccsd.CCSD'> does not have attributes  converged
E(CCSD) = -100.1005986095529  E_corr = -0.1412774422713259
CCSD(T) correction = -0.000977545349152716
e_ccsd_t error -0.0009775453491527163
```



