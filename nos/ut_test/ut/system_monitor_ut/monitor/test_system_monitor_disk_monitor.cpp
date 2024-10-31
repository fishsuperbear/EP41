// #define private public
// #define protected public

#include <thread>
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/monitor/system_monitor_disk_monitor.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorDiskMonitor : public testing::Test
{
    protected:
        void SetUp() {
            SystemMonitorConfig::getInstance()->LoadSystemMonitorConfig();
            SystemMonitorConfigInfo configInfo = SystemMonitorConfig::getInstance()->GetSystemMonitorConfigInfo();
            instance = new SystemMonitorDiskMonitor(configInfo.subFunction[SystemMonitorSubFunctionId::kDiskMonitor]);
        }

        void TearDown() {
            if (nullptr != instance) {
                delete instance;
                instance = nullptr;
            }
        }

    protected:
        SystemMonitorDiskMonitor* instance;
};

TEST_F(TestSystemMonitorDiskMonitor, Start)
{
    if (nullptr != instance) {
        instance->Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


TEST_F(TestSystemMonitorDiskMonitor, Stop)
{
    if (nullptr != instance) {
        instance->Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}