// HelloWorldClient.cpp
#include <unistd.h>

#include <iostream>
#include <string>
#include <functional>

#include "diag/dosomeip/api/dosomeip_transport.h"
#include "diag/dosomeip/common/dosomeip_def.h"
#include "diag/dosomeip/log/dosomeip_logger.h"
#include "diag/dosomeip/someip/src/DoSomeIPFuntions.h"


using namespace hozon::netaos::diag;

std::string getString(const DOSOMEIP_REGISTER_STATUS& status)
{
    switch (status)
    {
        case DOSOMEIP_REGISTER_STATUS::DOSOMEIP_REGISTER_STATUS_OK:
            return "DOSOMEIP_REGISTER_STATUS_OK";
        case DOSOMEIP_REGISTER_STATUS::DOSOMEIP_REGISTER_STATUS_TIMEOUT:
            return "DOSOMEIP_REGISTER_STATUS_TIMEOUT";
        default:
            return "NULL";
    }
}

void UDS_Req_CallBack(const DoSomeIPReqUdsMessage& msg)
{
    DS_INFO << "UDS_Req_CallBack: I am TPL, received a UDS req:";
    PrintRequest(msg);
}

void Register_CallBack(const DOSOMEIP_REGISTER_STATUS& status)
{
    DS_INFO << "Register_CallBack, status is :" << getString(status);
}

void UDS_Req_Send_Resp_Immediately_CallBack(const DoSomeIPReqUdsMessage& msg)
{
    DS_INFO << "UDS_Req_Send_Resp_Immediately_CallBack: I am TPL, received a UDS req:";
    PrintRequest(msg);
    DoSomeIPRespUdsMessage respMeg{};
    respMeg.udsSa = 0x8888;
    respMeg.udsTa = 0x6666;
    respMeg.result = 0;
    respMeg.taType = TargetAddressType::kFunctional;
    respMeg.udsData = {3,1,0,1,0,1,2,3,4,5,6,7,8,9};

    auto resp = DoSomeIPTransport::getInstance()->ReplyUdsOnSomeIp(respMeg);
    if(resp != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_02_connect init error.";
        return;
    }
}

void UDS_Req_Send_Resp_TimeOut_CallBack(const DoSomeIPReqUdsMessage& msg)
{
    DS_INFO << "UDS_Req_Send_Resp_TimeOut_CallBack: I am TPL, received a UDS req:";
    PrintRequest(msg);
    
    DoSomeIPRespUdsMessage respMeg{};
    respMeg.udsSa = 0x8888;
    respMeg.udsTa = 0x6666;
    respMeg.result = 0;
    respMeg.taType = TargetAddressType::kFunctional;
    respMeg.udsData = {3,1,0,1,0,1,2,3,4,5,6,7,8,9};
    std::this_thread::sleep_for(std::chrono::seconds(10));
    DS_DEBUG << "start set Resp Message !!!!";
    auto resp = DoSomeIPTransport::getInstance()->ReplyUdsOnSomeIp(respMeg);
    if(resp != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_02_connect init error.";
        return;
    }
}

// case 01 连接服务器
void TEST_01_connect()
{
    DS_INFO << "TEST_01_connect start.";

    auto res = DoSomeIPTransport::getInstance()->DosomeipInit(
        std::bind(&UDS_Req_CallBack, std::placeholders::_1),
        std::bind(&Register_CallBack, std::placeholders::_1)
    );

    if (res != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_01_connect init error.";
        return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(30));

    DoSomeIPTransport::getInstance()->DosomeipDeinit();
    DS_INFO << "TEST_01_connect end.";
}

// case 02 正常UDS响应（在超时时间内）
void TEST_02_connect()
{
    DS_INFO << "TEST_02_connect start.";

    auto res = DoSomeIPTransport::getInstance()->DosomeipInit(
        std::bind(&UDS_Req_Send_Resp_Immediately_CallBack, std::placeholders::_1),
        std::bind(&Register_CallBack, std::placeholders::_1)
    );

    if (res != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_02_connect init error.";
        return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(60));

    DoSomeIPTransport::getInstance()->DosomeipDeinit();
    DS_INFO << "TEST_02_connect end.";
}

// case 03 超时UDS响应
void TEST_03_connect()
{
    DS_INFO << "TEST_03_connect start.";

    auto res = DoSomeIPTransport::getInstance()->DosomeipInit(
        std::bind(&UDS_Req_CallBack, std::placeholders::_1),
        std::bind(&Register_CallBack, std::placeholders::_1)
    );

    if (res != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_03_connect init error.";
        return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(6));
    DoSomeIPRespUdsMessage respMeg{};
    respMeg.udsSa = 0x8888;
    respMeg.udsTa = 0x6666;
    respMeg.result = 0;
    respMeg.taType = TargetAddressType::kFunctional;
    respMeg.udsData = {3,1,0,1,0,1,2,3,4,5,6,7,8,9};
    std::this_thread::sleep_for(std::chrono::seconds(10));
    DS_DEBUG << "start set Resp Message !!!!";
    auto resp = DoSomeIPTransport::getInstance()->ReplyUdsOnSomeIp(respMeg);
    if(resp != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_02_connect init error.";
        return;
    }

    DoSomeIPTransport::getInstance()->DosomeipDeinit();
    DS_INFO << "TEST_03_connect end.";
}

// case 04 不响应
void TEST_04_connect()
{
    DS_INFO << "TEST_04_connect start.";

    auto res = DoSomeIPTransport::getInstance()->DosomeipInit(
        std::bind(&UDS_Req_CallBack, std::placeholders::_1),
        std::bind(&Register_CallBack, std::placeholders::_1)
    );

    if (res != DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK)
    {
        DS_ERROR << "TEST_04_connect init error.";
        return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1000));

    DoSomeIPTransport::getInstance()->DosomeipDeinit();
    DS_INFO << "TEST_04_connect end.";
}

void printHelp()
{
    std::cout << R"(
        用法: 
            ./someip_server [参数]      

        参数:
        test_01			连接服务器
        test_02			正常UDS响应(在超时时间内)
        test_03         超时UDS响应
        test_04         不响应

        举例：
        ./someip_server test01
    )" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please check argv. refer to the following help: " << std::endl;
        printHelp();
        return -1;
    }
    DoSomeIPLogger::GetInstance().SetLogLevel(
        static_cast<int32_t>(DoSomeIPLogger::DoSomeIPLogLevelType::DOSOMEIP_TRACE)
    );

    DoSomeIPLogger::GetInstance().InitLogging(
        "SOMEIP_SERVER",       // the id of application
        "SOMEIP_SERVER test",  // the log id of application
        DoSomeIPLogger::DoSomeIPLogLevelType::DOSOMEIP_TRACE,  // the log level of application
        hozon::netaos::log::HZ_LOG2CONSOLE |
            hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
        "./",  // the log file directory, active when output log to file
        10,    // the max number log file , active when output log to file
        20     // the max size of each  log file , active when output log to file
    );
    DoSomeIPLogger::GetInstance().CreateLogger("SOMEIP_SERVER");

    std::string msg = argv[1];

    if (msg == "test_01") {

        std::thread* thread01 = new std::thread(TEST_01_connect);
        thread01->join();
    } else if (msg == "test_02") {
        std::thread* thread02 = new std::thread(TEST_02_connect);
        thread02->join();
    } else if (msg == "test_03") {
        std::thread* thread03 = new std::thread(TEST_03_connect);
        thread03->join();
    } else if (msg == "test_04") {
        std::thread* thread04 = new std::thread(TEST_04_connect);
        thread04->join();
    } else {
        printHelp();
    }

    return 0;
}
