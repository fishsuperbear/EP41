#include <sstream>
#define private public
#define protected public
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <memory>
#include "gtest/gtest.h"
#include "log/include/logging.h"
#include "system_monitor/include/monitor/system_monitor_mnand_hs_monitor.h"


/***************测试说明***************/
/*
测试场景：测试系统监控中MNAND_HS监控结果获取是否正常
期望结果: 客户端可以正常获取到监控结果
*/

using namespace hozon::netaos::system_monitor;

class FuncTest:public ::testing::Test {

protected:
    static void SetUpTestSuite() {
        std::cout << "=== SetUpTestSuite ===" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "=== TearDownTestSuite ===" << std::endl;
    }

    void SetUp() override {
        system("export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH");
        SystemMonitorConfig::getInstance()->LoadSystemMonitorConfig();
        SystemMonitorConfigInfo configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
        m_instance = std::make_unique<SystemMonitorMnandHsMonitor>(configInfo.subFunction[SystemMonitorSubFunctionId::kEmmcMonitor]);
    }

    void TearDown() override {}

protected:
    std::unique_ptr<SystemMonitorMnandHsMonitor> m_instance;
};

TEST_F(FuncTest, GetUfsStatus) {
    std::vector<MnandHsInfo> ufsStatus;
    m_instance->GetUfsStatus(ufsStatus);
    uint res = ufsStatus.size();
    EXPECT_TRUE(0 != res);
}

TEST_F(FuncTest, GetEmmcStatus) {
    std::vector<MnandHsInfo> emmcStatus;
    m_instance->GetEmmcStatus(emmcStatus);
    uint res = emmcStatus.size();
    EXPECT_TRUE(0 != res);
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("system_monitor_mnand_hs_unit_test","utest",hozon::netaos::log::LogLevel::kOff,
                                hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
    hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}