# pyscf入门

## 安装pyscf

pyscf的文档网站已经给出了比较详细的安装教程,我直接选择了最简单的方法pip install 一个编译好的二进制文件,需要注意的是:

!!! tip "注意"

    pyscf 只支持在linux系统内安装,所以可以先装一个虚拟机或者开启wsl选项(适用于linux的windows子系统)

然后直接在linux终端输入:

```
pip install --prefer-binary pyscf
```

当然前提是要在linux系统中安装python,可以通过执行以下指令安装python:

```
sudo apt install python3 python3-pip python3-dev build-essential libhdf5-dev libopenblas-dev liblapack-dev libfftw3-dev libblas-dev
```

好了,这样就成功安装了pyscf,如果pip下来的不是最新版,可以通过以下指令更新:

```
pip install --upgrade pyscf
```

显示包信息:

```
pip show <package-name>
```
这样我们就可以获得这个包的全部信息,包括安装位置和作者

```
Name: pyscf
Version: 2.6.2
Summary: PySCF: Python-based Simulations of Chemistry Framework
Home-page: 
Author: 
Author-email: Qiming Sun <osirpt.sun@gmail.com>
License: Apache-2.0
Location: /usr/local/lib/python3.10/dist-packages
Requires: h5py, numpy, scipy, setuptools
Required-by: 
```

可以先进入python然后import一下看看有没有安装成功:

```
root@LAPTOP-DPOO7MIJ:/usr/local/lib/python3.10/dist-packages/pyscf# python3
Python 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pyscf
>>> exit()
```
我这里运行过程没有报错,看起来是安装成功了,当时安装的是python3,所以进入解释器的时候应该也要输python3,输python显示没有指令,报错了.

## 示例文件

因为第一次接触这个软件,所以完全不懂他的语法,可以看github上pyscf项目中的example文件夹,里面给出了大量的不同方法的计算实例,链接为:

[https://github.com/pyscf/pyscf/blob/master/examples/scf/00-simple_hf.py]

这个链接是hf方法的一个例子.

## 架构

无论是pyscf还是别的什么量化计算软件,其基础架构和要求都是差不多的,也就是说,他需要什么,能干什么,得到什么都是大同小异的,虽然不同的软件在使用不同计算方法的时候,其表现并不一定一致.

**首先,他需要一个输入文件** ,以单点能计算为例,我们需要计算给定结构下分子的能量,这就要求我们输入原子坐标,通常输入的是x-y-z坐标,以ORCA的输入文件为例:

```
! HF def2-TZVP

%scf
   convergence tight
end

* xyz 0 1
C  0.0  0.0  0.0
O  0.0  0.0  1.13
*
```

当然,上述的输入方式会存在一定的缺点,在几何构型已知的情况下,我们并不需要那么多自由度去表述这个分子,例如N个原子就需要3N个坐标来描述,在有些情况下,输入可以由键长和键角来描述,以水为例,只要输入两个键长一个键角,这就是Z-matrix输入

第二则是你要采用什么样的方法,在输入文件或者初始化部分,需要告诉计算程序你想要进行计算的近似方法如HF,DFT等等

**第三就是要明确计算任务**

例如,给定几何构型算分子能量,这就是单点能计算

给定分子组成,寻找能量最低的几何构型,这就是结构优化.

给定一个分子结构,计算其振动的频率,进一步可以进行稳定性分析和光谱分析,这就是频率计算.

还有什么偶极矩计算,g-factor计算等等许许多多的计算任务.

**第四就是要制定计算的基组** ,以pyscf为例,他用basis方法确定整个计算过程的基组:

```
mol.basis = 'sto-3g'
```

这个`sto-3g`是基组的名称,我也不懂是什么玩意

## pyscf的基础语法

### example-0

首先先看示例文件中给出的简单的HF方法计算HF(氟化氢)的能量:

```python
import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)

myhf = mol.HF()
myhf.kernel()

# Orbital energies, Mulliken population etc.
myhf.analyze()
```

首先,pyscf会提供一个个模块,例如`gto`,`scf`,模块里面会包含着大量的类,例如`gto.Mole`类会提供一个分子对象的模板,通过这个类可以创建一个分子,在上述代码中被简写为`pyscf.M`

根据示例文件中提供的另一种写法,我们可以先导入模块再定义对象:

```python
from pyscf import gto
mol = got.M(...)
```

在定义完成对象之后就相当于输入的初始化结束了,接下来就是要进行计算,例如采用mol对象中的Hartree-fork方法创建一个新的对象:

PS:这里使用的是`gto.M`方法,其已经隐式的调用了`mol.build()`进行初始化,所以不需要再初始化,事实上`gto.M`是`gto.Mole`的简化版本,所有mol的参数均可以在()内设定,但是根据`gto.Mole`方法创建的对象每次改变属性的时候都需要重新声明对象.

```python
myhf=mol.HF()
# 或者直接调用模块scf中的HF方法创建这个对象
myhf=scf.HF(mol)
```

最后一步就是调用方法启动内核进行计算:

```python
myhf.kernel()
```

最后再来看mol对象的配置部分,里面有许多参数是需要我们设置的.

原子坐标设置:

```python
atom='''
H 0 0 0
F 0 0 1.1
'''
```
在pyscf的文档网站中,说明了支持三引号表示坐标输入的例子,如果写在同一行的话,需要用`;`来进行间隔,在一个原子的不同坐标间,可以使用空格,也可以使用`,`来实现间隔

```python
atom='H 0. 0. 0;F 0. 0. 1.1'
```

字符串不区分大小写,同时还支持使用核电荷数代表原子进行输入:

```python
atom-'1 0 0 0;f 0 0 1.1'
```

`basis`参数设定的是基组,这个我也不知道取什么基组好,就按照文档中模仿,但是这个参数支持我们对不同的原子轨道使用不同的基组,这样可以实现计算资源的调度(不重要的分子用简单的基组):

```python
mol.basis = {'O': 'sto-3g', 'H': '6-31g'}
```

参数设定的最后一个

```python
symmetry = True
```

这个选项决定是否采用分子对称性加速计算.

计算完毕后,所有的信息会储存在myhf这个对象中,然后调用:

```python
myhf.analyze()
```
对相应的结果进行分析,如果没有这个就只会给你输出一个总能量,最后的结果为:

```
converged SCF energy = -99.987397440349
**** MO energy ****
MO #1   energy= -26.280057441301   occ= 2
MO #2   energy= -1.53365219237906  occ= 2
MO #3   energy= -0.67859798921924  occ= 2
MO #4   energy= -0.613428984030327 occ= 2
MO #5   energy= -0.613428984030317 occ= 2
MO #6   energy= 0.137118146058542  occ= 0
MO #7   energy= 0.673695200583408  occ= 0
MO #8   energy= 1.35287508928649   occ= 0
MO #9   energy= 1.35287508928649   occ= 0
MO #10  energy= 1.48918080509387   occ= 0
MO #11  energy= 1.6283291996278    occ= 0
MO #12  energy= 1.6283291996278    occ= 0
MO #13  energy= 2.02124864674874   occ= 0
MO #14  energy= 2.22702155259618   occ= 0
MO #15  energy= 4.0203151191834    occ= 0
MO #16  energy= 4.0203151191834    occ= 0
MO #17  energy= 4.13945032265318   occ= 0
MO #18  energy= 4.13945032265318   occ= 0
MO #19  energy= 4.78402661578955   occ= 0
 ** Mulliken atomic charges  **
charge of    0H =      0.40385
charge of    1F =     -0.40385
Dipole moment(X, Y, Z, Debye):  0.00000, -0.00000, -2.30763
```

第一行是总能量,下面的MO是分子轨道能量,occ代表占据分子轨道的电子个数.还给了原子的带电信息以及偶极矩,在z方向为-2.3德拜.

通过调节`verbose`这个参数(0-9),可以控制计算程序的输出,例如,设置:

```python
mol.verbose=4
```

就可以得到迭代的过程信息(我只取了一部分):

```
init E= -99.6496954142627
  HOMO = -0.640524605109698  LUMO = 0.0409210320076834
cycle= 1 E= -99.9414815223545  delta_E= -0.292  |g|= 0.469  |ddm|= 0.849
  HOMO = -0.499059037023119  LUMO = 0.123766658142965
cycle= 2 E= -99.9748833410617  delta_E= -0.0334  |g|= 0.29  |ddm|= 0.409
  HOMO = -0.606022039874914  LUMO = 0.124525299560854
cycle= 3 E= -99.9869447147443  delta_E= -0.0121  |g|= 0.03  |ddm|= 0.138
  HOMO = -0.607862046550919  LUMO = 0.135894361866866
```

## 一些零碎的设置

这部分内容是从pyscf手册上抄录整理的

### 电荷和自旋的设定

`charge`这个属性设置的是分子的带电量,默认情况`charge=0`,是不带电的,通过`charge`参数的设定,比较不同情况下分子的能量,可以得到电离能和电子亲和能.

通过对自旋的设定,可以计算单线态和三线态分子的能量的高低,从而验证电子排布是否符合洪特规则.其数值大小为alpha电子个数-beta电子的个数,也就是未配对电子个数

```python
mol.charge=1
mol.spin=1
```

==**注意改电荷的时候一定不要忘记改自旋多重度!!!不然输出变化有时候很不明显!!!**==

### 核的相关设置

考虑到可能会涉及到同位素的计算(这个对能量有贡献吗),所以可以对核的质量有设定:

```python
mol.mass={'O1':18;'H':2}
```
在一些考虑精确量子效应的场合中,核模型不再是点电荷,而是一个电荷云,其电荷密度满足高斯分布,所以需要设定:

```python
mol.nucmod = {'O1': 1} # nuclear charge model: 0-point charge, 1-Gaussian distribution
```

### 调用一些积分

这个暂时不是很懂,这些积分应该是手动实现计算方法的时候构建哈密顿量需要调用的:

```python
kin = mol.intor('int1e_kin')
vnuc = mol.intor('int1e_nuc')
overlap = mol.intor('int1e_ovlp')
eri = mol.intor('int2e')
```

### 坐标的相关设置

pyscf支持使用python的列表和数组以及numpy数组来生成atom坐标:

```python
mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H',(0, 0, 1)]]
mol.atom = (('O',numpy.zeros(3)), ['H', 0, 1, 0], ['H',[0, 0, 1]])
```

以及可以读入.xyz文件的坐标来构建分子:

```python
mol = gto.M(atom="my_molecule.xyz")
```

