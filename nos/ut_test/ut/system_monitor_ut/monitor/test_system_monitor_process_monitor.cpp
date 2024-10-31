// #define private public
// #define protected public

#include <thread>
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/monitor/system_monitor_process_monitor.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorProcessMonitor : public testing::Test
{
    protected:
        void SetUp() {
            SystemMonitorConfig::getInstance()->LoadSystemMonitorConfig();
            SystemMonitorConfigInfo configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
            instance = new SystemMonitorProcessMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kProcessMonitor]);
        }

        void TearDown() {
            if (nullptr != instance) {
                delete instance;
                instance = nullptr;
            }
        }

    protected:
        SystemMonitorProcessMonitor* instance;
};

TEST_F(TestSystemMonitorProcessMonitor, Start)
{
    if (nullptr != instance) {
        instance->Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


TEST_F(TestSystemMonitorProcessMonitor, Stop)
{
    if (nullptr != instance) {
        instance->Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}