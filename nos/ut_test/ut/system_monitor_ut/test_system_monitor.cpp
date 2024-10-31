// #define private public
// #define protected public

#include <thread>
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/system_monitor.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitor : public testing::Test
{
    protected:
        void SetUp() {
            instance = new SystemMonitor();
        }

        void TearDown() {
            if (nullptr != instance) {
                delete instance;
                instance = nullptr;
            }
        }

    protected:
        SystemMonitor* instance;
};

TEST_F(TestSystemMonitor, Init)
{
    if (nullptr != instance) {
        instance->Init();
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
}

TEST_F(TestSystemMonitor, Stop)
{
}

TEST_F(TestSystemMonitor, Run)
{
    if (nullptr != instance) {
        instance->Stop();
        instance->Run();
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
}

TEST_F(TestSystemMonitor, DeInit)
{
}