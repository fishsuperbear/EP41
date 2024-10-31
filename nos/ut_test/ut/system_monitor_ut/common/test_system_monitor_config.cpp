// #define private public
// #define protected public

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/common/system_monitor_config.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorConfig : public testing::Test
{
    protected:
        void SetUp() {
            instance = SystemMonitorConfig::getInstance();
        }

        void TearDown() {}

    protected:
        SystemMonitorConfig* instance;
};

TEST_F(TestSystemMonitorConfig, Init)
{
    if (nullptr != instance) {
        instance->Init();
    }
}

TEST_F(TestSystemMonitorConfig, LoadSystemMonitorConfig)
{
    if (nullptr != instance) {
        instance->LoadSystemMonitorConfig();
    }
}

TEST_F(TestSystemMonitorConfig, IsDiskMonitorPathList)
{
    if (nullptr != instance) {
        instance->IsDiskMonitorPathList("");
    }
}

TEST_F(TestSystemMonitorConfig, QueryPrintConfigData)
{
    if (nullptr != instance) {
        instance->QueryPrintConfigData();
    }
}

TEST_F(TestSystemMonitorConfig, DeInit)
{
    if (nullptr != instance) {
        instance->DeInit();
    }
}