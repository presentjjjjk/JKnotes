# 1.虚拟变量回归
## 1.1定义
在回购模型当中，除了定量变量之外，有时还需要通过引入一些不可量化的变量来刻画回归模型。例如：在考虑职工收入时，需要考虑到职工的受教育程度和性别；在研究冷饮店的需求量或者某个旅游胜地的旅游人数时，需要引入季节因素。这些非量化的因素或是变量称之为虚拟变量，或称为分类变量。

我们往往用$D$表示虚拟变量（用于强调其与别的定量变量之间的区别）；通常，虚拟变量为只取0或1的人工变量——基础类型或是肯定类型取值为1；比较类型或是否定类型取值为0——往往作为解释变量使用。

## 1.2回归中的应用
事实上，虚拟变量的定义与使用方法与我们在非线性回归当中对于定性变量的使用是相同的。从而我们可以使用一些非线性回归的手段对其进行分析——即分类讨论的手法