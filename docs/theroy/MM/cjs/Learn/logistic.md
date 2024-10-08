# logistic回归
为了解决非线性回归当中的部分问题，提出此著名的逻辑斯蒂回归模型，其形式为：

$$
f(x)=\frac{e^x}{1+e^x}=\frac{1}{1+e^{-x}}
$$

事实上这就是Sigmoid函数，其值域是（0,1）。

我们在之前的非线性回归当中说到过$E(y)=\pi$，事实上还有一种理解的方式就是$y_i=1$在总体当中的比例。运用这个思路，我们可以使用$y=1$的比例来代替因变量$y$本身做回归，==这样的话一定程度上就可以解决因变量是离散变量带来的一系列问题。==

## 1.1分组logistic回归
研究一个买房的问题。我们知道买房子的人如果按照年收入分类，其实可以分为很多很多类，如果我们要研究不同类中的买房子的人占来售楼处看房子的人的比例，那么应该如何处理呢？

我们考虑去拟合一个每组都使用的模型：

$$
p_i=\frac{exp(\beta_0+\beta_1x_i)}{1+exp(\beta_0+\beta_1x_i)},\quad i=1,2,\dots,c
$$

其中c为分组数据的组数，$x_i$为对应组的家庭年收入。对此，我们只需设：

$$
p_i^,=ln(\frac{p_i}{1-p_i})
$$

就可以得到：

$$
p_i^,=\beta_0+\beta_1x_1+\epsilon
$$

到此为止，我们算是解决了离散性和值域的问题，但是异方差性还没有解决。

那么此时我们可以考虑加权最小二乘法。相关性告诉我们，当$n_i$较大的时候，$p_i^,$的近似方差为为$\frac{1}{m_i\pi_i(1-\pi_i)}$。因此把$\pi_i$用估计值代入，可以得到我们的权值：

$$
\omega_i=m_ip_i(1-p_i)
$$

其中每一个$n_i,p_i$，分别为对应组的值个数和观测比例。

## 1.2未分组logistic回归
上述情况相当于是在分组情况下的回归。那么我们需要考虑未分组数据的logistic回归模型。

所谓未分组，就是只有一个组，那么就只有一个定性因变量和一系列对应的自变量。上面使用了加权最小二乘估计来拟合模型，但是这个地方只有一个组的$y=1$的比例的数据（因为这里的样本的因变量都是定型变量值，并且没有分组），所以就不能够使用上面的方法拟合参数了。这里我们使用极大似然估计来估计参数。

设$y$的均值为$\pi$，并且我们假设有一组观测值（$x_{i1},\dots,x_{ip};y_i$）。由此假设，我们在一般情况下，结合回归的定义，可以得到一组关系;

$$
E(y_i)=\pi_i=f(\beta_0+\beta_1x_{i1}+\dots+\beta_px{ip})
$$

由于这种模型规定的值域，一般认为这里的$f$是一个一元连续性随机向量的分布函数。不同的分布函数就可以得到不用的模型。logistic的回归模型也算其中一个，但代入后会得到一个比较长的式子

由此，相当于每一个$y_i$服从了一个均值为$\pi_i$的伯努利分布。那么其概率还是可以写成：

$$
P(y_i)=\pi_i^{y_i}(1-\pi_i)^{1-y_i}
$$

那么代入每一个样本，就可以得到其似然函数：

$$
L=\displaystyle\Pi^n_{i=1}\pi_i^{y_i}(1-\pi_i)^{1-y_i}\\ lnL=\displaystyle\sum^n_{i=1}[y_iln\frac{\pi_i}{1-\pi_i}+ln(1-\pi_i)]
$$

我们刚刚说的要把logistic回归的函数形式代入，即：

$$
\pi_i=\frac{exp(\beta_0+\beta_1x_{i1}+\dots+\beta_px_{ip})}{1+exp(\beta_0+\beta_1x_{i1}+\dots+\beta_px_{ip})}
$$

代入上述对数表达式，结合计算数值方法进行估计（使得值最大）
## 1.3多类别logistic回归

此回归类型，有k个类别的因变量，对应的数值记为：$1,2,\dots,k$。类比之前因变量是0-1变量的情形，我们可以考虑对其用回归模型，计算其落在每一组的概率。

给定样本数据$(x_{i1},\dots,x_{ip};y_i),i=1,2,\dots,n$,那么显然要拟合在不同组的概率，我们就需要不同的回归系数，所以多类别logistic回归模型先定义为：

$$
\pi_{ij}=\frac{exp(\beta_{0j}+\beta_{1j}x_{i1}+\dots+\beta_{pj}x_{ip})}{exp(\beta_{01}+\beta_{11}x_{i1}+\dots+\beta_{p1}x_{ip})+\dots +exp(\beta_{0k}+\beta_{1k}x_{i1}+\dots+\beta_{pk}x_{ip})}
$$


这事实上就是类比了sigmoid函数（$e^0=1$）。此处表示第i组样本的因变量$y_i$取第j个类别的概率，注意分母中每一项对应的自变量列表都是给定的那一组自变量。

但是这个模型是有问题的。当我们给每个回归系数增加一个常数时，我们得到的$\pi_{ij}$是不变的。所以为了破坏这种齐次性，我们把分母的第一项的系数全部人工设为0，那么模型就变成了；

$$
\pi_{ij}=\frac{exp(\beta_{0j}+\beta_{1j}x_{i1}+\dots+\beta_{pj}x_{ip})}{1+exp(\beta_{02}+\beta_{12}x_{i1}+\dots+\beta_{p2}x_{ip})+\dots +exp(\beta_{0k}+\beta_{1k}x_{i1}+\dots+\beta_{pk}x_{ip})}
$$

从而形成了与之前sigmoid函数形式相同的情况。事实上，这就是：

$$
ln\frac{P(y_i=j)}{P(y_i=1)}=\beta_{0j}+\beta_{1j}x_{i1}+\dots+\beta_{pj}x_{ip}
$$

当然了，因为它还是属于未分组类型的Logistic回归，所以我们还是要使用极大似然估计和数值方法来解决这些问题。其对应的似然函数为：

$$
L(\beta|y)=\frac{\Pi^n_{i=1}\Pi^q_{j=1}exp\left\{(\beta_{0j}+\beta_{1j}x_{i1}+\dots+\beta_{pj}x_{ip})I_{\left\{y_i=j\right\}}\right\}}{[1+\sum^n_{l=1}exp\left\{\beta_{0l}+x_{i1}\beta_{1l}+\dots+x_{ip}\beta_{pl}\right\}]^n}
$$

其中$I\left\{y_i=j\right\}$是其分布的特征函数，即当大括号内式子成立时函数值为1，不成立时函数值为0

## 1.4因变量是顺序数据的回归模型

顺序变量的意思就是，我们把每一种绩点的情况对应一下，分为几类，60-的为第一类，60-63的为第二类，64-67的为第三类，依次往上。可以看出，给定了一些范围后，“分数”这个连续性的变量就可以通过这样的规定转为了一个离散型的定性变量。这种变量我们叫顺序变量。

用数学的方式来说，就是认为一个连续型随机变量$Y^*$和$p-1$个门限值$\theta_1<\dots<\theta_{p-1},$对应关系为：

$$
Y=l\Leftarrow\ \Rightarrow \theta_{l-1}\leq Y^*<\theta_l
$$

并约定$\theta_0=-\infty,\theta_p=+\infty$

我们这里添加一个线性模型的假设，即连续型随机变量,即连续型随机变量$Y^*$服从一个线性模型，如果有线性性的假设，我们就可以把模型写成$Y^*=x^,\beta+\epsilon$的形式

这里有个地方需要强调一下，我们的残差不一定再符合正态化的条件了。所以这种情况我们需要人为假定它的密度函数，即需要假设：

$$
\epsilon \sim f(\epsilon)
$$

那么这个时候有

$$
\epsilon=Y^*-X\beta\sim f(y-X\beta)
$$

这里的$f$是我们假设的$\epsilon$的密度函数的形式。积分可得：

$$
P(Y=l)=\displaystyle\int^{\theta_l}_{\theta_{l-1}}f(y^*-x\beta)dy^*=F(\theta^l-x^,\beta)-F(\theta^{l-1}-x^,\beta)
$$

因为$x\beta$这一项不是随机变量，所以不会影响$y^*$的分布函数形式，因此还是可以这么积分的

那么如果要对这样的模型做回归，也是一样需要极大似然估计的，对应的似然函数为

$$
L(\theta,\beta)=\Pi_{i=1}^n\left\{F(\theta_{y_i}-x_i^,\beta)-F(\theta_{y_{i-1}}-x_i^,\beta)\right\}
$$

其中$\theta=(\theta_1,\dots,\theta_{k-1})$

$Y^*$本身是一个连续型的随机变量，也可以通过线性模型去拟合。但是我们更多的是希望关注它落到某一类的概率。从而根据上面推导，我们需要$Y^*$的密度函数或者分布函数，不然没有办法做极大似然估计。

但我们已知$\epsilon$的密度函数或者分布函数，所以就可以根据它得到$Y^*$的密度函数，而 $Y^*$的密度函数的要求又正好符合我们之前讨论的Logistic回归的内容，所以这样的话，就有一些显式的表达式代入到上面，可以一定程度上简化问题。

