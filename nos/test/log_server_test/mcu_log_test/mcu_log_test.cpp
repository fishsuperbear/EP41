#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include "logging.h"

using namespace hozon::netaos::log;

//简单测试所写的封装
int main(int argc, char* argv[]) {

    InitLogging("APP_COMPRESS",                                                             // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kDebug,                                                      //the log level of application
                                    HZ_LOG2FILE,  //the output log mode
                                    "/log/",                                                                  //the log file directory, active when output log to file
                                    20,                                                                    //the max number log file , active when output log to file
                                    10                                                                     //the max size of each  log file , active when output log to file
    );
    auto testlog6{CreateLogger("TEST", "hozon log test3", LogLevel::kDebug)};
    int count {0};
    double dTestlValue = 1234.5678;

    while (count <= 20) {
        testlog6->LogDebug() << "double = " << dTestlValue;
        count++;
    }

    std::string appId{"MCU"};
    std::string ctxId{"POWER"};

    InitMcuLogging(appId);
    auto log = CreateMcuLogger(appId, ctxId);
    log->LogDebug() << "mcu log";

    std::string appId2{"DESAY"};
    std::string ctxId2{"VOL"};

    InitMcuLogging(appId2);
    auto log2 = CreateMcuLogger(appId2, ctxId2);
    auto log3 = CreateMcuLogger(appId2, ctxId);

    log2->LogDebug() << "desay log";
    log3->LogDebug() << "desay log power mode";
    
    return 0;
}