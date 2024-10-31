#include "logging.h"
#include <thread>
#include <stdio.h>
#include <iostream>
#include <signal.h>

using namespace hozon::netaos::log;

uint8_t stopFlag = 0;
std::thread* pLogCTX1_TestThread1 = nullptr;
std::thread* pLogCTX1_TestThread2 = nullptr;
std::thread* pLogCTX1_TestThread3 = nullptr;

void SigHandler(int signum)
{
    std::cout << "--- log test sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
}

void LogCTX1_TestThread()
{
    // CreateLogger and get a logger
    auto testlog1{hozon::netaos::log::CreateLogger("TEST_CTX1", "hozon log test1",
                                                hozon::netaos::log::LogLevel::kDebug)};
    
    while(!stopFlag)
    {
        testlog1->LogDebug() << "testlog Debug" << 1;
        testlog1->LogWarn() << "testlog Warn" << 1;
        testlog1->LogError() << "testlog Error" << 2;

        testlog1->LogDebug() << "testlog Debug" << 1;
        testlog1->LogWarn() << "testlog Warn" << 1;
        testlog1->LogError() << "testlog Error" << 2;

        bool bBollValue = false;
        double dTestlValue = 1234.5678;
        float fTestlValue = 1234.5678;
        testlog1->LogCritical() << "bBollValue = " << bBollValue;

        std::uint32_t ui32TestlValue = 100000;
        testlog1->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog1->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog1->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog1->LogInfo() << "fTestlValue in float = " << fTestlValue;

        testlog1->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog1->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog1->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog1->LogInfo() << "fTestlValue in float = " << fTestlValue;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    

}

void LogCTX2_TestThread()
{
    auto testlog2{hozon::netaos::log::CreateLogger("TEST_CTX_2", "hozon log test2",
                                                hozon::netaos::log::LogLevel::kInfo)};

    while(!stopFlag)
    {
        testlog2->LogDebug() << "testlog Debug" << 1;
        testlog2->LogWarn() << "testlog Warn" << 1;
        testlog2->LogError() << "testlog Error" << 2;

        testlog2->LogDebug() << "testlog Debug" << 1;
        testlog2->LogWarn() << "testlog Warn" << 1;
        testlog2->LogError() << "testlog Error" << 2;

        bool bBollValue = false;
        double dTestlValue = 1234.5678;
        float fTestlValue = 1234.5678;
        testlog2->LogCritical() << "bBollValue = " << bBollValue;

        std::uint32_t ui32TestlValue = 100000;
        testlog2->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog2->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog2->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog2->LogInfo() << "fTestlValue in float = " << fTestlValue;

        testlog2->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog2->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog2->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog2->LogInfo() << "fTestlValue in float = " << fTestlValue;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    
}

void LogCTX3_TestThread()
{
    auto testlog3{hozon::netaos::log::CreateLogger("TEST_CTX3", "hozon log test3",
                                                hozon::netaos::log::LogLevel::kInfo)};
    // auto operation_log_app = hozon::netaos::log::GetOperationLogger(hozon::netaos::log::OperationLogType::tApp);
    while(!stopFlag)
    {
        testlog3->LogDebug() << "testlog Debug" << 1;
        testlog3->LogWarn() << "testlog Warn" << 1;
        testlog3->LogError() << "testlog Error" << 2;

        testlog3->LogDebug() << "testlog Debug" << 1;
        testlog3->LogWarn() << "testlog Warn" << 1;
        testlog3->LogError() << "testlog Error" << 2;

        bool bBollValue = false;
        double dTestlValue = 1234.5678;
        float fTestlValue = 1234.5678;
        testlog3->LogCritical() << "bBollValue = " << bBollValue;

        // operation_log_app->LogCritical() << "bBollValue = " << bBollValue;

        std::uint32_t ui32TestlValue = 100000;
        testlog3->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog3->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        // operation_log_app->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        // operation_log_app->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog3->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog3->LogInfo() << "fTestlValue in float = " << fTestlValue;

        testlog3->LogInfo() << "ui32TestlValue in dec = " << ui32TestlValue;
        testlog3->LogInfo() << "ui32TestlValue in hex = 0x" << LogHex32{ui32TestlValue};

        testlog3->LogInfo() << "dTestlValue in double = " << dTestlValue;
        testlog3->LogInfo() << "fTestlValue in float = " << fTestlValue;

        
        

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

//简单测试所写的封装
int main(int argc, char * argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    // hozon::netaos::log::InitLogging(
    //     "TEST_IF",
    //     "hozon log test Application",
    //     LogLevel::kError,
    //     HZ_LOG2CONSOLE,
    //     "./",
    //     10,
    //     (20 * 1024 * 1024)
    // );

    hozon::netaos::log::InitLogging("./etc/log_cfg.json");

    
    

    

    pLogCTX1_TestThread1 = new std::thread(LogCTX1_TestThread);
    pLogCTX1_TestThread2 = new std::thread(LogCTX2_TestThread);
    pLogCTX1_TestThread3 = new std::thread(LogCTX3_TestThread);
    std::string ctxID = "ctx01";
    LogLevel level = LogLevel::kDebug;

    while (!stopFlag) {
        auto operation_log_app = CreateOperationLogger(ctxID, "xxx", level);
        operation_log_app->LogCritical() << "Critical log of reserved";
        // OK
        operation_log_app->LogError() << "Critical log of Error";
        // OK
        operation_log_app->LogWarn() << "Critical log of Warn";
        // OK
        operation_log_app->LogInfo() << "Critical log of Info";
        // OK
        operation_log_app->LogDebug() << "Critical log of Debug";
        // NOT_OK
        operation_log_app->LogTrace() << "Critical log of Trace";

        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }

    return 0;
}