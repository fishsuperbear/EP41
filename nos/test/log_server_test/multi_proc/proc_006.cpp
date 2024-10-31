#include "logging.h"
#include <thread>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace hozon::netaos::log;

int main(int argc, char * argv[])
{
    hozon::netaos::log::InitLogging(
        "Proc006",
        "hozon log test Application",
        LogLevel::kDebug,
        HZ_LOG2CONSOLE | HZ_LOG2FILE,
        "/log/",
        10,
        (20 * 1024 * 1024)
    );

    std::string ctxID1 = "ctx01";
    std::string ctxID2 = "ctx02";
    std::string ctxID3 = "ctx03";

    LogLevel level = LogLevel::kDebug;
    std::string message = "message";

    auto operation_log_app = CreateOperationLogger(ctxID1, "xxx", level);
    auto operation_log_app2 = CreateOperationLogger(ctxID2, "xxx", level);

    auto operation_log_app3 = CreateLogger(ctxID3, "xxx", level);

    operation_log_app->LogWarn() << message;
    operation_log_app2->LogWarn() << message;
    operation_log_app3->LogError() << "I AM log.";

    return 0;
}