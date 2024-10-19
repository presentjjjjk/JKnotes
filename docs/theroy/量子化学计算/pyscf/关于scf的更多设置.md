# 关于scf的更多设置

github上提供了很多的example,下面的内容是我从示例代码中整理的.

## 初始矩阵密度的扰动

样例上给了两种计算方法,一种是直接调用复杂方法`GHF`进行计算,另一种则是对初始密度矩阵进行微小复数扰动后再调用`GHF`方法进行计算:

```python

# 复数GHF计算
mf = scf.GHF(mol)
dm = mf.get_init_guess() + 0j
dm[0, :] += .05j
dm[:, 0] -= .05j

mf.kernel(dm0=dm)  # 执行自洽场计算
```

这个是`H2O+`分子,但是这个体系比较简单,两种方法的计算结果几乎一样,但是时间是复数那个来的久:

```
converged SCF energy = -75.3515062334015  <S^2> = 2.7517054  2S+1 = 3.4650861
实数GHF计算时间: 3.6811秒
converged SCF energy = -75.3515051124689  <S^2> = 2.7780301  2S+1 = 3.4802472
复数GHF计算时间: 5.6591秒
```

## level-shift

在HF 计算过程中,如果自旋轨道之间的能级间隙过于接近,那么可能会导致算法无法正确收敛,以及在某些体系中,alpha和beta轨道并不是简并的,这个时候就需要对自旋做一定的偏移使得体系更好的进行收敛

这个我暂时没有很深的体会,可能之后会体会到.

```python

# Break alpha and beta degeneracy by applying different level shift
#
mf = scf.UHF(mol)
mf.level_shift = (0.3, 0.2)
mf.kernel()
```

## 双激发计算和无收缩基组

`DHF`即考虑了双重激发的HF计算方法,有一部分的电子关联

以计算HCL为例,考虑电子激发可以获得更精确的能量.

同时,对于一些复杂且重要的分子如H-CL中的Cl原子,我们可以考虑给原子指定不同的基组来加速计算,例如,指定Cl的基组为无收缩的ccpvdz基组,指定H的基组为普通的ccpvdz基组.

这两种基组的差别在于,收缩基组是固定好几个高斯函数前面的系数,将他们组合成为原子轨道后作为原子轨道基组,也就是说,在总组合中,这几个高斯函数的系数是一起变化的.

而无收缩基组就不固定高斯函数前的系数,在收敛迭代的过程中确定系数,这样的描述显然会更加精确:

pyscf中使用字典来制定原子轨道基组:

```
basis = {'Cl': 'unc-ccpvdz',
             'H' : 'ccpvdz'},
```

使用无收缩基组的时间成本会提升

```
converged SCF energy = -461.421612153775
计算时间:7.438036680221558
converged SCF energy = -463.392784627854
计算时间:10.732428073883057
```

## 更加复杂的相互作用

在pyscf中可以开启一些自旋效应,相对论效应引起的相互作用,文档里面给出的示例是Breit和Gaunt相互作用,完全不懂是什么意思

通过设置参数开启:

```python
mf.with_gaunt = True
mf.with_breit = True
```

计算结果和实验符合的很好:

```
converged SCF energy = -461.443187882259
E(Dirac-Coulomb) = -461.443187882259, ref = -461.443188093533
converged SCF energy = -461.326149590612
E(Dirac-Coulomb-Gaunt) = -461.326149590612, ref = -461.326149787363
converged SCF energy = -461.334922568572
E(Dirac-Coulomb-Breit) = -461.334922568572, ref = -461.334922770344
```

## 弥散函数

在描述长程相互作用的时候,比如氢键或者一些带负电很多的负离子,我们需要在基组中加入弥散函数,这个时候就要使用拓展基组:

```python
mol.basis = 'aug-ccpvdz'
```

## 快速牛顿法

这是一种比默认方法要快的优化方法,通常在构建自洽场对象的时候,在外面再套一层`scf.fast_newton()`:

```python
mf=scf.fast_newton(scf.RHF(mol))
```

但是经过试验,发现计算小分子的时候,还是默认方法更快,例如,下面是HCl的单点能计算结果:

```
converged SCF energy = -459.989656978441
快速牛顿法用时 2.2400104999542236
converged SCF energy = -459.989656978441
默认方法用时 1.632215976715088
```

然后尝试一个比较大的体系(双氯联苯)

```
converged SCF energy = -1378.11231899653
快速牛顿法用时 76.15665197372437
converged SCF energy = -1378.11231899655
默认方法用时 98.44448351860046
```

可以看到快速牛顿法有着显著的优势,值得注意的是`scf.fast_newton`就已经执行运算了,不需要在下面再加一个`mf.kernel()`,否则会重复计算,浪费计算资源.


## 改进初始猜测

很多时候,我们的初始猜测是软件从头开始生成的,但是如果我们要做大量的类似的重复的计算,每次都从头开始猜测肯定是不妥当的,这样会浪费很多的计算资源.所以可以让HF跑几圈,然后将生成的系数矩阵作为更好的初始猜测保存下来,这样在每一次调用类似的计算的时候就能节省时间.


```python
mf = scf.RHF(mol)
mf.chkfile = 'cu3-diis.chk'# 检查点文件,用于保存少量迭代的结果
mf.max_cycle = 2 # 设定HF的最大迭代次数为2,用于改进初始猜测
mf.kernel()

#
# Load the DIIS results then use it as initial guess for function
# scf.fast_newton
#
mf = scf.RHF(mol)
mf.__dict__.update(scf.chkfile.load('cu3-diis.chk', 'scf'))
scf.fast_newton(mf)
```