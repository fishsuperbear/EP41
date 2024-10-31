// #define private public
// #define protected public

#include <thread>
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/monitor/system_monitor_network_monitor.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorNetworkMonitor : public testing::Test
{
    protected:
        void SetUp() {
            SystemMonitorConfig::getInstance()->LoadSystemMonitorConfig();
            SystemMonitorConfigInfo configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
            instance = new SystemMonitorNetworkMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kNetworkMonitor]);
        }

        void TearDown() {
            if (nullptr != instance) {
                delete instance;
                instance = nullptr;
            }
        }

    protected:
        SystemMonitorNetworkMonitor* instance;
};

TEST_F(TestSystemMonitorNetworkMonitor, Start)
{
    if (nullptr != instance) {
        instance->Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


TEST_F(TestSystemMonitorNetworkMonitor, Stop)
{
    if (nullptr != instance) {
        instance->Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}