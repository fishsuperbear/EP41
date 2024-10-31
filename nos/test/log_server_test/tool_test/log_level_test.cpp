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
        "APP01",
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

    // 单个普通日志
    {
        auto logger1 = CreateLogger(ctxID1, "LOG_CTX1", level_error);
        logger1->LogTrace() << "loglevel : Trace, needLogout : FALSE";
        logger1->LogDebug () << "loglevel : Debug, needLogout : FALSE";
        logger1->LogInfo() << "loglevel : Info, needLogout : FALSE";
        logger1->LogWarn() << "loglevel : Warn, needLogout : FALSE";
        logger1->LogError() << "loglevel : error, needLogout : TRUE";
        logger1->LogCritical() << "loglevel : Critical, needLogout : TRUE";
    }

    // 单个普通日志
    {
        auto logger2 = CreateLogger(ctxID2, "LOG_CTX2", level_debug);
        logger2->LogTrace() << "loglevel : Trace, needLogout : FALSE";
        logger2->LogDebug () << "loglevel : Debug, needLogout : FALSE";
        logger2->LogInfo() << "loglevel : Info, needLogout : TRUE";
        logger2->LogWarn() << "loglevel : Warn, needLogout : TRUE";
        logger2->LogError() << "loglevel : error, needLogout : TRUE";
        logger2->LogCritical() << "loglevel : Critical, needLogout : TRUE";
    }

    // 单个OP日志
    {
        auto op_logger1 = CreateOperationLogger(ctxID3, "OP_LOG_CTX3", level_error);
        op_logger1->LogTrace() << "OP loglevel : Trace, needLogout : FALSE";
        op_logger1->LogDebug () << "OP loglevel : Debug, needLogout : FALSE";
        op_logger1->LogInfo() << "OP loglevel : Info, needLogout : FALSE";
        op_logger1->LogWarn() << "OP loglevel : Warn, needLogout : FALSE";
        op_logger1->LogError() << "OP loglevel : error, needLogout : TRUE";
        op_logger1->LogCritical() << "OP loglevel : Critical, needLogout : TRUE";
    }

    // case1 ： 直接输出
    // 输出 ：
    // [2023-07-25 10:29:25.210] [erro] [LOG_CTX1] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 10:29:25.210] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    // case2 ：设置所有CTX为Off等级 
    // export HZ_SET_LOG_LEVEL=IGNORE.IGNORE:kOff

    // case3 ： 设置LOG_CTX1为Critical等级  
    // cmd : export HZ_SET_LOG_LEVEL=IGNORE.LOG_CTX1:kCritical
    // 输出 ：
    // [2023-07-25 10:56:17.712] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 10:56:17.712] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    // case3 ： 设置OP_LOG_CTX3为Critical等级  
    // cmd : export HZ_SET_LOG_LEVEL=IGNORE.OP_LOG_CTX3:kCritical
    // 输出 ：
    // [2023-07-25 11:01:03.696] [erro] [LOG_CTX1] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 11:01:03.696] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    // case4 : 设置APP等级为Critical
    // cmd : export HZ_SET_LOG_LEVEL=APP01.IGNORE:kCritical
    // 输出 ：
    // [2023-07-25 11:03:16.417] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 11:03:16.417] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 11:03:16.417] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]

    // case5: 设置爱APP 且 LOG_CTX1为 Off 等级 
    // cmd : export HZ_SET_LOG_LEVEL=APP01.LOG_CTX1:kCritical
    // 输出 ：
    // [2023-07-25 13:16:50.733] [crit] [LOG_CTX1] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [info] [LOG_CTX2] [loglevel : Info, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [warn] [LOG_CTX2] [loglevel : Warn, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [erro] [LOG_CTX2] [loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [crit] [LOG_CTX2] [loglevel : Critical, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [erro] [operation] [OP_LOG_CTX3] [OP loglevel : error, needLogout : TRUE]
    // [2023-07-25 13:16:50.733] [crit] [operation] [OP_LOG_CTX3] [OP loglevel : Critical, needLogout : TRUE]


    return 0;
}