这个文件夹下的代码是所有video模块的公用代码，里面只会包含platform层的一些宏，
并不包含video模块的具体类别和具体型号。
当然，可以定义一些ops来给具体的模块来实现。


下面这个commit号，运行multigroup和ipc producer/consumer(使能了later attach跑单路)项目运行ok：
commit e3111998ca887134b72373ee62cd2ca6d2c7ef2d (HEAD -> dev_v0.1, origin/dev_v0.1)
Author: Wangjinfeng <wangjingfeng@hozonauto.com>
Date:   Fri Feb 24 02:54:02 2023 +0000

    Fix the failure of the producer to bind the socket twice.

