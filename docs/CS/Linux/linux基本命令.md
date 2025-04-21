# linux基本命令

linux的操作指令主要就是针对文件的操作指令,多用就好了

## 文件操作

- `cd`命令主要用于切换目录,可以`cd 绝对路径`来进入某个我想要进入的文件夹,也可以使用相对路径:`cd ..`是返回上一级目录,`cd ~`是返回当前用户的家目录,`cd -`是返回上一次所在的目录

```
cd /homework/CC   # 注意,linux路径之间使用正斜杠来分隔路径的
```

- `ls`命令用于列出当前目录下的所有文件:

```
root@LAPTOP-DPOO7MIJ:~/homework# ls 
CC  CI  DFT  HF自洽场  MP2
```

- `pwd`用于显示当前目录所在的绝对路径:

```
root@LAPTOP-DPOO7MIJ:~/homework# pwd
/root/homework
```

- `mkdir`指令用于在当前目录下创建一个文件夹:

```
root@LAPTOP-DPOO7MIJ:~/homework# mkdir test
root@LAPTOP-DPOO7MIJ:~/homework# ls
CC  CI  DFT  HF自洽场  MP2  test
```

- `cp`和`mv`两个指令分别对应windows系统中的复制粘贴和剪切,直接`cp 文件名1 文件名2`就会在当前目录下复制文件1并且将它重命名为文件2:

```
root@LAPTOP-DPOO7MIJ:~/homework/test# cp test.py test1.py
root@LAPTOP-DPOO7MIJ:~/homework/test# ls
test.py  test1.py
```

或者可以直接指定绝对路径:

```
root@LAPTOP-DPOO7MIJ:~/homework/test# cp /root/homework/CC/CCtry.py test3.py
root@LAPTOP-DPOO7MIJ:~/homework/test# ls
test.py  test1.py  test3.py
```

但是注意要从根目录开始写绝对路径,否则会提示找不到文件.如果当前目录已经存在同名文件,则会直接重写覆盖.

- `mv`命令用于移动文件,在同一目录下执行相当于重命名:

```
root@LAPTOP-DPOO7MIJ:~/homework/test# mv test3.py test4.py
root@LAPTOP-DPOO7MIJ:~/homework/test# ls
test.py  test1.py  test4.py
```

在不同目录下执行则相当于剪切,我们可以使用两个绝对路径,或者其中一个使用相对路径:

```
root@LAPTOP-DPOO7MIJ:~/homework/test# mv test4.py ..
root@LAPTOP-DPOO7MIJ:~/homework/test# cd ..
root@LAPTOP-DPOO7MIJ:~/homework# ls
CC  CI  DFT  HF自洽场  MP2  test  test4.py
```

- `rm`命令代表删除操作,谨慎使用:

```
root@LAPTOP-DPOO7MIJ:~/homework# rm test4.py
root@LAPTOP-DPOO7MIJ:~/homework# ls
CC  CI  DFT  HF自洽场  MP2  test
```

在删除文件夹的时候不能直接用,要在中间加一个`-r`:`root@LAPTOP-DPOO7MIJ:~/homework# rm -r test`,包括`cp`和`mv`命令对文件夹操作的时候也一样,需要在中间加一个`-r`

### 编辑文件

上面的指令没有涉及到创建一个新文件以及编辑他,可以使用linux自带的编辑器vim来编辑文件:

以下是一些基本命令：

1. 打开文件：
   ```
   vim 文件名
   ```

2. 模式切换：
   - 按 `i` 进入插入模式
   - 按 `Esc` 返回普通模式

3. 保存和退出：
   - `:w` 保存文件
   - `:q` 退出（如果文件未修改）
   - `:wq` 保存并退出
   - `:q!` 强制退出不保存

4. 移动光标：
   - `h`（左）、`j`（下）、`k`（上）、`l`（右）
   - `0` 移到行首，`$` 移到行尾
   - `gg` 移到文件开头，`G` 移到文件末尾

5. 编辑操作：
   - `x` 删除当前字符
   - `dd` 删除当前行
   - `yy` 复制当前行
   - `p` 粘贴

6. 搜索：
   - `/关键词` 向下搜索
   - `?关键词` 向上搜索
   - `n` 下一个匹配项，`N` 上一个匹配项

7. 撤销和重做：
   - `u` 撤销上一步操作
   - `Ctrl + r` 重做

这些是vim的基本命令，熟练使用可以大大提高文本编辑效率。



