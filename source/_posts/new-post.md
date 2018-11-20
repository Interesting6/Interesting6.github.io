---
title: new-post
date: 2018-02-27 14:29:07
tags: test
---


# this is a test blog.
welcome to treamy's world.


### Create a new post
``` bash
$ hexo new "My New Post"
```


### add some code
``` python
>>> print("My New Post")
```


### 标签页面

1>运行以下命令
``` bash
$ hexo new page "tags"
```

同时，在/source目录下会生成一个tags文件夹，里面包含一个index.md文件


### 推送到服务器上
``` bash
$ hexo g -d
```
先generate一下生成静态页面，再deploy部署到服务器。
好像`hexo d -g`也可以。。一次记错了写成这个也行。。我也不知道为啥在一个网页上看到说他们左右是相同的，还是用前面那个较好解读的吧。

