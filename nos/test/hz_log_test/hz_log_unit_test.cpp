#include "logging.h"
#include <thread>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace hozon::netaos::log;

// enum class LogLevel : uint8_t {
//     kOff = 0x00U,
//     kFatal = 0x01U,
//     kError = 0x02U,
//     kWarn = 0x03U,
//     kInfo = 0x04U,
//     kDebug = 0x05U,
//     kVerbose = 0x06U
// };

enum initType
{
    kFile       = 0,
    kConsole    = 1,
    kBoth       = 2,
};

void init(initType type)
{
    if (type == initType::kFile)
    {
        hozon::netaos::log::InitLogging("HZ_TEST_APP",                                                     // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kTrace,                                                      // the log level of application
                                    hozon::netaos::log::HZ_LOG2FILE,                                       // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
        );
    }
    if (type == initType::kConsole)
    {
        hozon::netaos::log::InitLogging("HZ_TEST_APP",                                                     // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kTrace,                                                      // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE,                                    // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
        );
    }
    if (type == initType::kBoth)
    {
        hozon::netaos::log::InitLogging("HZ_TEST_APP",                                                     // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kTrace,                                                      // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE + hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20                                                                     // the max size of each  log file , active when output log to file
        );
    }
}

void testCase01()
{
    init(initType::kBoth);
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    testlog->LogCritical() << "LogCritical";
    testlog->LogError() << "LogError";
    testlog->LogWarn() << "debug";
    testlog->LogInfo() << "LogInfo";
    testlog->LogDebug() << "LogDebug";
    testlog->LogTrace() << "LogTrace";
}

void testCase02()
{
    init(initType::kBoth);
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    auto testlog2 = hozon::netaos::log::CreateLogger("CTX2", "hozon log", hozon::netaos::log::LogLevel::kInfo);

    testlog->LogCritical() << "LogCritical";

    testlog2->LogCritical() << "LogCritical";
}

void testCase03()
{
    init(initType::kBoth);
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    int i = 0;
    while (1)
    {   
        if(i >= 1000000)
        {
            break;
        }
        testlog->LogError() << "LogError";
        i++;
    }
}

void testCase04()
{
    hozon::netaos::log::InitLogging("HZ_TEST_APP",                                                     // the id of application
                                    "HZ_log application",                                                  // the log id of application
                                    LogLevel::kTrace,                                                      // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE + hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                                    "./",                                                                  // the log file directory, active when output log to file
                                    10,                                                                    // the max number log file , active when output log to file
                                    20,                                                                     // the max size of each  log file , active when output log to file
                                    true
        );
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    testlog->LogCritical() << "LogCritical";
    testlog->LogError() << "LogError";
    testlog->LogWarn() << "debug";
    testlog->LogInfo() << "LogInfo";
    testlog->LogDebug() << "LogDebug";
    testlog->LogTrace() << "LogTrace";
}

void testCase05()
{
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    testlog->LogCritical() << "LogCritical";
    testlog->LogError() << "LogError";
    testlog->LogWarn() << "debug";
    testlog->LogInfo() << "LogInfo";
    testlog->LogDebug() << "LogDebug";
    testlog->LogTrace() << "LogTrace";
}

void testCase06()
{
    init(initType::kBoth);
    auto testlog = hozon::netaos::log::CreateLogger("CTX", "hozon log", hozon::netaos::log::LogLevel::kInfo);
    bool bBollValue = false;
    double dTestlValue = 1234.5678;
    float fTestlValue = 1234.5678;
    std::uint32_t ui32TestlValue = 100000;

    testlog->LogInfo() << "bBollValue = " << bBollValue;
    testlog->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
    testlog->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};
    testlog->LogInfo() << "dTestlValue in double = " << dTestlValue;
    testlog->LogInfo() << "fTestlValue in float = " << fTestlValue;
}

//简单测试所写的封装
int main(int argc, char * argv[])
{

    int test_case;
    std::cout << "Enter the test case number (1-6): ";
    std::cin >> test_case;

    switch (test_case) {
    case 1:
        std::thread(testCase01).join();
        break;
    case 2:
        std::thread(testCase02).join();
        break;
    case 3:
        std::thread(testCase03).join();
        break;
    case 4:
        std::thread(testCase04).join();
        break;
    case 5:
        std::thread(testCase05).join();
        break;
    case 6:
        std::thread(testCase06).join();
        break;
    default:
        std::cout << "Invalid test case number\n";
        break;
    }

    return 0;
}