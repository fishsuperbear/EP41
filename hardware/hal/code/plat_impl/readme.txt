这个文件夹下都是用来实现interface里定义的接口时会用到的哪些通用逻辑代码

这是为了定义一些在hardware_hal里可以使用的一些ops函数集，来做一些比如日志记录、分配内存之类的通用逻辑，这些逻辑可以单独实现来做成一个so，这样可以把与硬件相关联的逻辑的部分彻底与平台环境分离开，甚至还可以像android 8.0以后那样单独升级系统或者单独升级硬件抽象层的hal及其实现（只要这部分hw_platform的接口不变）