// #define private public
// #define protected public

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/handler/system_monitor_handler.h"

using namespace hozon::netaos::system_monitor;

class TestSystemMonitorHandler : public testing::Test
{
    protected:
        void SetUp() {
            instance = SystemMonitorHandler::getInstance();
        }

        void TearDown() {}

    protected:
        SystemMonitorHandler* instance;
};

TEST_F(TestSystemMonitorHandler, Init)
{
    if (nullptr != instance) {
        instance->Init();
    }
}

TEST_F(TestSystemMonitorHandler, NotifyEventSend)
{
    if (nullptr != instance) {
        instance->NotifyEventSend(SystemMonitorSubFunctionId::kTemperatureMonitor, "");
    }
}

TEST_F(TestSystemMonitorHandler, AlarmEventSend)
{
    if (nullptr != instance) {
        instance->AlarmEventSend(SystemMonitorSubFunctionId::kTemperatureMonitor, "");
    }
}

TEST_F(TestSystemMonitorHandler, ControlEventCallBack)
{
    SystemMonitorControlEventInfo info;
    info.id = SystemMonitorSubFunctionId::kTemperatureMonitor;
    info.type = SystemMonitorSubFunctionControlType::kMonitorSwitch;
    info.value = "off";
    if (nullptr != instance) {
        instance->ControlEventCallBack(info);
    }
}

TEST_F(TestSystemMonitorHandler, ReportFault)
{
    SystemMonitorSendFaultInfo info;
    info.faultId = 4050;
    info.faultObj = 1;
    info.faultStatus = 0;
    if (nullptr != instance) {
        instance->ReportFault(info);
    }
}

TEST_F(TestSystemMonitorHandler, DeInit)
{
    if (nullptr != instance) {
        instance->DeInit();
    }
}