[PART 00]
    本工具现在支持以下命令：
        用法: 
            ./hz_log_tools [参数] [log文件路径]      持续输出指定的日志文件
        或: ./hz_log_tools [参数] [日志Level信息]      修改日志等级

        参数:
        setLogLevel			设定等级(用法参考[PART 02])
        continue			持续输出日志 

        举例：
        1. ./hz_log_tools setLogLevel HZ_TEST.IGNORE:kError
        2. ./hz_log_tools continue /opt/usr/ytx/HZ_TEST_0000_2023-04-01_04-56-15.log

[PART 01]
    需要tab补全参数功能，按如下手顺操作：
        1.导入可执行程序路径，加入环境变量PATH
            例如：export PATH=$PATH:/opt/usr/mdc/tools/hz_log_tools
        2.执行complete.sh脚本
            例如: source complete.sh
        3.即可进行tab补全参数

[PART 02]

    日志Level信息可通过以下方式设置：

    一、通过“hz_set_log_level”可执行程序设置：
    1、编译生成可执行文件（x86为例）：netaos/output/x86/tools/hz_set_log_level
    2、运行：./hz_set_log_level appId.ctxId:level
        如：./hz_set_log_level TEST.TEST_CTX:kDebug，表示将appId为“TEST”， ctxId为“TEST_CTX”的log ctx的日志等级调整为“Debug”。

    二、通过环境变量“HZ_SET_LOG_LEVEL”设置：
    1、进程运行前，终端上输入：export HZ_SET_LOG_LEVEL=TEST.TEST_CTX:kDebug


    其他说明：
    1、若不设置指定“appId”的log等级， 则将“appId”设置成“IGNORE”，如：“IGNORE.TEST_CTX:kDebug”表示将所有ctxId为“TEST_CTX”的log ctx的log 等级设为“Debug”。
    2、若不设置指定“ctxId”的log等级， 则将“ctxId”设置成“IGNORE”，如：“TEST.IGNORE:kDebug”表示将appId为“TEST”的所有的log ctx的log 等级设为“Debug”。
    3、若“appId”与“ctxId”均设置成“IGNORE”，表示设置所有进程所有log ctx的log等级。
        如：“IGNORE.IGNORE:kInfo”表示所有进程所有log ctx的log等级设置为“Info”
    4、LOG等级只能为: "kCritical", "kError", "kWarn", "kInfo", "kDebug", "kTrace"中的一种

[PART 03]

    Operation Log 支持方式（MDC）：
    一、修改文件"/etc/syslog.conf"，在文件中添加：
    # log from HZ-TEST
    local0.*              /home/var/log/hz_app.log
    local1.*              /home/var/log/hz_mw.log
    local2.*              /home/var/log/hz_bsp.log
    local3.*              /home/var/log/hz_os.log
    local4.*              /home/var/log/hz_supplier.log
    local5.*              /home/var/log/hz_others.log
    local6.*              /home/var/log/hz_reserved.log

    二、使用时，调用接口“GetOperationLogger”，得到一个智能指针。
    调用接口时，根据需要传入相应的Operation Log类型，如下：
    enum class OperationLogType : uint8_t {
        tApp = 0x00U,
        tMw = 0x01U,
        tBsp = 0x02U,
        tOs = 0x03U,
        tSupplier = 0x04U,
        tOthers = 0x05U,
        tReserved = 0x06U
    };
    
    三、输出相应level 的log，例如：

        auto operation_log_bsp = hozon::netaos::log::GetOperationLogger(hozon::netaos::log::OperationLogType::tBsp);
        operation_log_bsp->LogError() << "Error log of bsp";

    四、在“/home/var/log/”中，查找相应文件，得到相应d额operation log，例如“/home/var/log/hz_bsp.log”：
        Mar 23 02:44:34 AOS local2.err [22285]: [hz_bsp] [Error log of bsp]
        Mar 23 02:44:39 AOS local2.err [22285]: [hz_bsp] [Error log of bsp]
        Mar 23 02:44:44 AOS local2.err [22285]: [hz_bsp] [Error log of bsp]
