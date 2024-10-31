关于C++接口：
1）接口文件路径：hal\code\interface\hpp\intf_camera\1.0
2）库文件路径：hal\code\lib

关于C++接口的demo参考：
1）关于gpu数据的使用范例参考：demo\camera_hpp_demo\camera_hpp_cuda
2）关于cpu数据的使用范例参考：demo\camera_hpp_demo\camera_hpp_enc
3）关于producer那一头的使用范例参考（中间件同学当前可以不用关注）：demo\camera_hpp_demo\camera_hpp_main

lowlevel C接口：
1）device&module框架相关（对接到上层C++接口的实现）：hal\hw_hal
2）设备ops定义：\hal\code\interface\devices
3）具体当前平台当前软件模式的项目：hal\code\modules\devices\video\modules\nvmedia
	nvmedia_multiipc_consumer_cuda和nvmedia_multiipc_consumer_enc


