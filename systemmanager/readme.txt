#动态修改log等级(需要root权限)
#1.insmod systemmanager.ko
#2.echo [level] > /proc/netahal/hallog_ctrl/global
#注：若已安装systemmanager模块跳过第一步，level范围1~5
#ERROR          5
#WARNING        4
#INFO           3
#TRACE          2
#DEBUG          1


