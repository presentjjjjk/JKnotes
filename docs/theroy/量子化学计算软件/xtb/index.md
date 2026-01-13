# xtb

!!! note "在线手册"

    https://xtb-docs.readthedocs.io/en/latest/

!!! note "安装方法"

    1.去https://github.com/grimme-lab/xtb/releases/tag/v6.7.1 下载最新版本的安装包,目前我使用的是xtb-6.7.1.

    2.tar -xf 解压对应的安装包然后移动到相应的目录下

    3.编辑~/bsshrc文件,指定相应的路径:

    ```
    export PATH=$PATH:/root/xtb/xtb-6.7.1/bin
    export XTBPATH=/root/xtb/xtb-6.7.1/share/xtb

    # 设置并行的核心数和每个线程使用的最大内存量
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OMP_STACKSIZE=1000M
    ulimit -s unlimited
    ```

    4.重启终端,即可使用xtb命令

