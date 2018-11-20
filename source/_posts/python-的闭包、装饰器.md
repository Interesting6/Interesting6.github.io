---
title: python 的闭包、装饰器
date: 2018-03-17 17:33:14
tags: [fun math, python]
categories: [math, python]
---


# 一、闭包（Closure）
## 什么是闭包？
> 在计算机科学中，闭包（英语：Closure），又称词法闭包（Lexical Closure）或函数闭包（function closures），是引用了自由变量的函数。这个被引用的自由变量将和这个函数一同存在，即使已经离开了创造它的环境也不例外。所以，有另一种说法认为闭包是由函数和与其相关的引用环境组合而成的实体。闭包在运行时可以有多个实例，不同的引用环境和相同的函数组合可以产生不同的实例。—— [维基百科](https://zh.wikipedia.org/wiki/闭包_(计算机科学))

这里我给个简单的解释：一个**闭包**就是你调用了一个*函数A*，这个*函数A*返回了一个*函数B*给你。这个**返回的函数B**就叫做闭包。你在**调用函数A**的时候**传递的参数**就是**自由变量**。

## 举个例子：
``` python
def func(name):
    def inner_func(age):
        print 'name:', name, 'age:', age
    return inner_func

bb = func('matrix')
bb(26)  # >>> name: matrix age: 26

```
这里面调用func的时候就产生了一个闭包-- *inner_func*,并且该闭包持有自由变量-- *name*，因此这也意味着，当函数func的生命周期结束之后，**name这个变量依然存在**，因为它被闭包引用了，所以不会被回收。
* 也有人说这种内部函数inner_func可以使用外部函数的变量name的行为就叫闭包。


# 二、装饰器（Decorator）
## 什么是装饰器？
> “装饰器的功能是将被装饰的函数当作参数传递给与装饰器对应的函数（名称相同的函数），并返回包装后的被装饰的函数”


听起来有点绕，没关系，直接看示意图,其中a为 与*装饰器@a*对应的函数，*b*为装饰器修饰的函数，*装饰器@a*的作用是：
![image](/images/decorator.png)
**简而言之：@a 就是将 b 传递给 a()，并返回新的 b = a(b)**

## 举个例子
1.  先导入包：
``` python
from functools import reduce
import math
import logging
logging.basicConfig(level=logging.INFO)
```
2. 定义一个检查参数的装饰器：
``` python
def checkParams(fn):
    def wrapper(*numbers):
        temp = map(lambda x:isinstance(x,(int,)),numbers) # 检查参数是否都为整型
        if reduce(lambda x,y: x and y, temp): # 若都为整型
            return fn(*numbers)             # 则调用fn(*numbers)返回计算结果
        #否则通过logging记录错误信息，并友好退出
        logging.warning("variable numbers cannot be added")
        return
    return wrapper     #fn引用gcd，被封存在闭包的执行环境中返回
```
3. 然后定义求最大公约数的函数（能求多个的最大公约数）：
``` python
def gcd(*numbers):
    """return the greatest common divisor of the given integers."""
    return reduce(math.gcd, numbers)
```
4. 调用
``` bash
>>>gcd = checkParams(gcd)
>>>gcd(3, 'hello')
# 输出 WARNING:root: variable numbers cannot be added
```
注意checkParams函数：
>* 首先看参数fn，当我们调用checkParams(gcd)的时候，它将成为函数对象gcd的一个本地(Local)引用；
>* 在checkParams内部，我们定义了一个wrapper函数，添加了参数类型检查的功能，然后调用了fn(*numbers)，根据LEGB法则，解释器将搜索几个作用域，并最终在(Enclosing层) checkParams函数的本地作用域中找到fn；
>* 注意最后的return wrapper，这将创建一个闭包，fn变量(gcd函数对象的一个引用)将会封存在闭包的执行环境中，不会随着checkParams的返回而被回收；

当调用gcd = checkParams(gcd)时，gcd指向了新的wrapper对象，它添加了参数检查和记录日志的功能，同时又能够通过封存的fn，继续调用原始的gcd进行最大公约数运算。

因此调用gcd(3, 'hello')将不会返回计算结果，而是打印出日志：root: variable numbers cannot be added。


有人觉得add = checkParams(add)这样的写法未免太过麻烦，于是python提供了一种更优雅的写法，被称为**语法糖**：
``` python
@checkParams
def lcm(*numbers):
    """return lowest common multiple."""
    f = lambda a,b:int((a*b)/gcd(a,b))
    return reduce(f, numbers)
```
其实这只是一种写法上的优化，解释器仍然会将它转化为gcd = checkParams(gcd)来执行。




