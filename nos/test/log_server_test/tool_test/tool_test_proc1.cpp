#include "logging.h"
#include <thread>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace hozon::netaos::log;

// enum class LogLevel : uint8_t {
//     kTrace = 0x00U,
//     kDebug = 0x01U,
//     kInfo = 0x02U,
//     kWarn = 0x03U,
//     kError = 0x04U,
//     kCritical = 0x05U,
//     kOff = 0x06U
// };

int main(int argc, char * argv[])
{
    hozon::netaos::log::InitLogging(
        "APP_Proc1",
        "hozon log test Application",
        LogLevel::kInfo,
        HZ_LOG2CONSOLE | HZ_LOG2FILE,
        "/log/",
        10,
        (20 * 1024 * 1024)
    );

    std::string ctxID1 = "LOG_CTX1";
    std::string ctxID2 = "LOG_CTX2";
    std::string ctxID3 = "OP_LOG_CTX3";

    LogLevel level_debug = LogLevel::kDebug;
    LogLevel level_error = LogLevel::kError;
    
    auto logger1 = CreateLogger(ctxID1, "LOG_CTX1", level_error);
    auto logger2 = CreateLogger(ctxID2, "LOG_CTX2", level_debug);
    auto op_logger1 = CreateOperationLogger(ctxID3, "OP_LOG_CTX3", level_error);

    for (size_t i = 0; i < 5; i++)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "waiting for change log level ..." << std::endl;
    }

    logger1->LogTrace() << "loglevel : Trace, needLogout : FALSE";
    logger1->LogDebug () << "loglevel : Debug, needLogout : FALSE";
    logger1->LogInfo() << "loglevel : Info, needLogout : FALSE";
    logger1->LogWarn() << "loglevel : Warn, needLogout : FALSE";
    logger1->LogError() << "loglevel : error, needLogout : TRUE";
    logger1->LogCritical() << "loglevel : Critical, needLogout : TRUE";
    
    logger2->LogTrace() << "loglevel : Trace, needLogout : FALSE";
    logger2->LogDebug () << "loglevel : Debug, needLogout : FALSE";
    logger2->LogInfo() << "loglevel : Info, needLogout : TRUE";
    logger2->LogWarn() << "loglevel : Warn, needLogout : TRUE";
    logger2->LogError() << "loglevel : error, needLogout : TRUE";
    logger2->LogCritical() << "loglevel : Critical, needLogout : TRUE";

    op_logger1->LogTrace() << "OP loglevel : Trace, needLogout : FALSE";
    op_logger1->LogDebug () << "OP loglevel : Debug, needLogout : FALSE";
    op_logger1->LogInfo() << "OP loglevel : Info, needLogout : FALSE";
    op_logger1->LogWarn() << "OP loglevel : Warn, needLogout : FALSE";
    op_logger1->LogError() << "OP loglevel : error, needLogout : TRUE";
    op_logger1->LogCritical() << "OP loglevel : Critical, needLogout : TRUE";
    
    // case 1 设置所有进程关闭日志
    // cmd : hz_log_tools setloglevel IGNORE.IGNORE:kOff

    // case 2 设置APP_Proc1日志等级为kCritical
    // cmd : hz_log_tools setloglevel APP_Proc1.IGNORE:kCritical
    // proc1 ：
    // 输出 ：
    // [2023-07-25 13:40:18.245] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:40:18.245] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:40:18.245] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]
    // proc2 ：
    // 输出 ：
    // [2023-07-25 13:40:20.437] [erro] [LOG_CTX1] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:40:20.437] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    // case 3 设置APP_Proc2，ctx为LOG_CTX2 的日志等级为kCritical
    // cmd : hz_log_tools setloglevel APP_Proc2.LOG_CTX2:kCritical
    // proc 1 :
    // 输出 ：
    // [2023-07-25 13:48:20.878] [erro] [LOG_CTX1] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:48:20.878] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]
    // proc 2 : 
    // 输出 ：
    // [2023-07-25 13:48:21.718] [erro] [LOG_CTX1] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:48:21.718] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:48:21.718] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:48:21.718] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:48:21.718] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    return 0;
}