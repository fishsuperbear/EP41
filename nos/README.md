# 编译环境
当前只验证过Ubuntu20.04

# 首次拉取代码
执行`./netaos/scripts/fetch_code.sh`，拉取三方库

# 构建说明
通过执行`./build.sh`完成构建，参数如下
* -p: 指定目标平台，支持x86/mdc/j5，默认为x86
* -t: 指定编译类型，支持release/debug，默认为release
* -c: 清除编译缓存，在切换不同平台时需要执行
* -h: 显示帮助信息
<br>例如，`./build.sh -p j5 -t debug`为编译j5平台的debug版本