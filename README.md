# 欢迎 
### 主要提供那些内容
这个仓库我用来存放一些自己的学习笔记,包括已经完成的课程和未完成的课程,笔记里面包含了大量的我主观推测的内容,仅供参考,不定时更新(真的会更新吗),笔记的网址为https://presentjjjjk.github.io/JKnotes/

### 怎么搭建一个和我一样的笔记?

本笔记基于一个python包(MKdocs)和meterial主题搭建,首先你要去python官网下载一个python https://www.python.org/ ,然后使用pip包管理器安装MKdocs和meterial主题,就可以开始在本地的书写啦.

在本地的笔记制作完毕后,我将它发布到了github上,并且依靠github-page托管我的网站,首先你要创建一个本地库和远程库,然后依次运行:

```
git add .
git commit -m "update"
git push origin master  # 或者是 git push origin main
```
上传到github后,执行下面指令将网站部署到github page上:

```
mkdocs gh-deploy # 这是mkdocs提供的一个命令，可以直接将网站部署到github pages上
```

最后就可以访问你的网站了.






