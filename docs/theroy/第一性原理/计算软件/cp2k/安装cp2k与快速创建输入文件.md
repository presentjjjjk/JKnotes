# 安装cp2k与快速创建输入文件

CP2k是一款免费开源的第一性原理计算软件,其计算速度快(比vasp快很多),使用方便(至少有了社长开发的Multiwfn之后使用很方便)

## 安装方法

可以直接看社长的安装方法[http://sobereva.com/586]

首先从github上获取安装包:[https://github.com/cp2k/cp2k/releases/]

然后使用tar-xf命令解压到本地目录,例如我是直接解压到root目录下,然后运行以下命令,以我下载的2024.3版本为例

```bash
cd /root/cp2k-2024.3/tools/toolchain/
./install_cp2k_toolchain.sh --with-sirius=no --with-openmpi=system --with-plumed=instal
```

with表示会自动安装这些依赖库,我这里已经装过openmpi库了,所以写`--with-openmpi=sysem`自动调用系统中的openmpi,注意你的机子上必须有gcc和gFortran,因为cp2k的很多库是依赖于c++和fortran写的,其次,你还需要确保有uzip和unrar这两个模块,方便安装了时候可以顺利完成自动解压.

下一步,把/root/cp2k-2024.3/tools/toolchain/install/arch/下所有文件拷到/root/cp2k-8.1/arch目录下。然后准备开始编译,编译之前还得装个库,由于我是Ubuntu用户,运行`sudo install zlib-dev`,然后运行:

```
source /root/cp2k-2024.3/tools/toolchain/install/setup
cd /sob/cp2k-2024.3
make -j 16 ARCH=local VERSION="ssmp psmp"
```

编译的时候有几个核就用几个,加快编译速度.然后配置系统环境变量:

```
source /root/cp2k-2024.3/tools/toolchain/install/setup
export PATH=$PATH:/root/cp2k-2024.3/exe/local
```

社长建议自己编译的用popt版运行,所以直接再加一句`alias cp2k='mpirun -np 8 cp2k.popt'`,这样直接就可以使用`cp2k`命令执行计算了,至此安装过程完毕.

## 晶体结构数据库

ICSD

[https://icsd.fiz-karlsruhe.de/index.xhtml;jsessionid=DD774556A382B8D7D36B2FABB422D7F7]

可以直接搜索晶体下载cif文件

或者看这个链接 [https://blog.shishiruqi.com//2019/05/01/resources/] 里面有丰富的第一性原理学习资料.

## 使用multiwfn创建cp2k输入文件

详情参考社长的文章[http://sobereva.com/587]

首先你需要一个cif文件,这个大概是从晶体结构数据库下载过来的或者用vesta画的,获得含有晶胞信息的输入文件后,直接拖入到Multiwfn中:

```
cp2k # 进入cp2k模块
\enter 
选择一些设置,例如k点网格个数,使用的基组和泛函
0 # 直接生成输入文件
```

功能列表如下:

```
 -11 Enter the interface for geometry operations
 -10 Return
 -9 Other settings
 -7 Set direction(s) of applying periodic boundary condition, current: XYZ
 -4 Calculate atomic charges, current: None
 -3 Set exporting cube file, current: None
 -2 Toggle exporting .molden file for Multiwfn, current: No
 -1 Choose task, current: Energy
  0 Generate input file now!
  1 Choose theoretical method, current: PBE
  2 Choose basis set and pseudopotential, current: DZVP-MOLOPT-SR-GTH
  3 Set dispersion correction, current: None
  4 Switching between diagonalization and OT, current: Diagonalization
  5 Set density matrix mixing, current: Broyden mixing
  6 Toggle smearing electron occupation, current: No
  7 Toggle using self-consistent continuum solvation (SCCS), current: No
  8 Set k-points, current: GAMMA only
 15 Toggle calculating excited states via TDDFT, current: No
```