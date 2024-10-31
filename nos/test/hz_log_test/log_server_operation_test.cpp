#include "logging.h"
#include <thread>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace hozon::netaos::log;

uint8_t stopFlag = 0;

void SigHandler(int signum)
{
    std::cout << "--- log test sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
}

int main(int argc, char * argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    hozon::netaos::log::InitLogging(
        "TEST1111",
        "hozon log test Application",
        LogLevel::kDebug,
        HZ_LOG2CONSOLE | HZ_LOG2FILE,
        "./",
        10,
        (20 * 1024 * 1024)
    );

    std::string appID = "app01";
    std::string ctxID = "ctx01";
    std::string ctxID2 = "ctx02";
    std::string ctxID3 = "ctx03";

    LogLevel level = LogLevel::kDebug;
    std::string message = "message";

    auto operation_log_app = CreateOperationLogger(ctxID, "xxx", level);
    auto operation_log_app2 = CreateOperationLogger(ctxID2, "xxx", level);

    auto operation_log_app3 = CreateLogger(ctxID3, "xxx", level);
    operation_log_app3->LogError() << "I AM log.";

    auto operation_log_app4 = CreateLogger(ctxID, "xxx", level);
    operation_log_app4->LogError() << "I AM log 2222.";

    // // OK
    operation_log_app->LogCritical() << message; 
    // // OK
    operation_log_app->LogError() << message;
    // NOT OK
    operation_log_app->LogWarn() << message;
    operation_log_app2->LogWarn() << message;

    // // OK
    operation_log_app->LogInfo() << message;
    // // OK
    operation_log_app->LogDebug() << message;
    // // NOT_OK
    operation_log_app->LogTrace() << message;

    return 0;
}