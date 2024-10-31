// #define private public
// #define protected public

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/manager/system_monitor_manager.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorManager : public testing::Test
{
    protected:
        void SetUp() {
            instance = SystemMonitorManager::getInstance();
        }

        void TearDown() {}

    protected:
        SystemMonitorManager* instance;
};

TEST_F(TestSystemMonitorManager, Init)
{
    if (nullptr != instance) {
        instance->Init();
    }
}

TEST_F(TestSystemMonitorManager, ControlEvent)
{
    SystemMonitorControlEventInfo info;
    info.id = SystemMonitorSubFunctionId::kAllMonitor;
    info.type = SystemMonitorSubFunctionControlType::kMonitorSwitch;
    info.value = "off";
    if (nullptr != instance) {
        instance->ControlEvent(info);
    }
}

TEST_F(TestSystemMonitorManager, DeInit)
{
    if (nullptr != instance) {
        instance->DeInit();
    }
}