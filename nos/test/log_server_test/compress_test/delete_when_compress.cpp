#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include "logging.h"

using namespace hozon::netaos::log;

std::uint32_t GetTickCount() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

//简单测试所写的封装
int main(int argc, char* argv[]) {



    hozon::netaos::log::InitLogging("APP_COMPRESS",                                                             // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kDebug,                                                      //the log level of application
                                    hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                    "/log/",                                                                  //the log file directory, active when output log to file
                                    20,                                                                    //the max number log file , active when output log to file
                                    10                                                                     //the max size of each  log file , active when output log to file
    );
    auto testlog6{hozon::netaos::log::CreateLogger("TEST", "hozon log test3", hozon::netaos::log::LogLevel::kDebug)};

    std::uint32_t start_time = GetTickCount();
    std::uint64_t count{0};
    double dTestlValue = 1234.5678;

    while (count <= 200000) {
        testlog6->LogDebug() << "double = " << dTestlValue;

        count++;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));

    }
    std::uint32_t end_time = GetTickCount();
    std::cout << "1000000's ,Total time is : " << 0.001 * (end_time - start_time);

    return 0;
}