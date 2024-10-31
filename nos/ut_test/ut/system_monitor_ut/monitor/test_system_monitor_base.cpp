// #define private public
// #define protected public

#include <thread>
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/monitor/system_monitor_base.h"

using namespace hozon::netaos::system_monitor;

class TestMonitorBase : public SystemMonitorBase {
public:
    TestMonitorBase(const SystemMonitorSubFunctionInfo& funcInfo)
    : SystemMonitorBase(funcInfo, "temp_monitor.log") {}
    virtual ~TestMonitorBase() {}

    virtual void Start() {}
    virtual void Stop() {}
};

class TestSystemMonitorBase : public testing::Test
{
    protected:
        void SetUp() {
            SystemMonitorSubFunctionInfo info;
            info.shortName = "TemperatureMonitor";
            info.id = SystemMonitorSubFunctionId::kTemperatureMonitor;
            info.monitorSwitch = "on";
            info.monitorCycle = 3000;
            info.recordFileCycle = 10;
            info.recordFilePath = "/log/";
            info.isAlarm = true;
            info.alarmValue = 60;
            instance = new TestMonitorBase(info);
        }

        void TearDown() {
            if (nullptr != instance) {
                delete instance;
                instance = nullptr;
            }
        }

    protected:
        TestMonitorBase* instance;
};

TEST_F(TestSystemMonitorBase, GetMonitorName)
{
    std::string name = "";
    if (nullptr != instance) {
        name = instance->GetMonitorName();
    }

    EXPECT_EQ(name, "TemperatureMonitor");
}

TEST_F(TestSystemMonitorBase, GetMonitorId)
{
    SystemMonitorSubFunctionId id;
    if (nullptr != instance) {
        id = instance->GetMonitorId();
    }

    EXPECT_EQ(id, SystemMonitorSubFunctionId::kTemperatureMonitor);
}

TEST_F(TestSystemMonitorBase, GetMonitorSwitch)
{
    std::string sw = "";
    if (nullptr != instance) {
        sw = instance->GetMonitorSwitch();
    }

    EXPECT_EQ(sw, "on");
}

TEST_F(TestSystemMonitorBase, SetMonitorSwitch)
{
    std::string sw = "";
    if (nullptr != instance) {
        instance->SetMonitorSwitch("off");
        sw = instance->GetMonitorSwitch();
    }

    EXPECT_EQ(sw, "off");
}

TEST_F(TestSystemMonitorBase, GetMonitorCycle)
{
    int cycle = 0;
    if (nullptr != instance) {
        cycle = instance->GetMonitorCycle();
    }

    EXPECT_EQ(cycle, 3000);
}

TEST_F(TestSystemMonitorBase, SetMonitorCycle)
{
    int cycle = 0;
    if (nullptr != instance) {
        instance->SetMonitorCycle(5000);
        cycle = instance->GetMonitorCycle();
    }

    EXPECT_EQ(cycle, 5000);
}

TEST_F(TestSystemMonitorBase, GetRecordFileCycle)
{
    int cycle = 0;
    if (nullptr != instance) {
        cycle = instance->GetRecordFileCycle();
    }

    EXPECT_EQ(cycle, 10);
}

TEST_F(TestSystemMonitorBase, SetRecordFileCycle)
{
    int cycle = 0;
    if (nullptr != instance) {
        instance->SetRecordFileCycle(15);
        cycle = instance->GetRecordFileCycle();
    }

    EXPECT_EQ(cycle, 15);
}

TEST_F(TestSystemMonitorBase, GetRecordFilePath)
{
    std::string path = "";
    if (nullptr != instance) {
        path = instance->GetRecordFilePath();
    }

    EXPECT_EQ(path, "/log/");
}

TEST_F(TestSystemMonitorBase, SetRecordFilePath)
{
    std::string path = "";
    if (nullptr != instance) {
        instance->SetRecordFilePath("/log/system_monitor/");
        path = instance->GetRecordFilePath();
    }

    EXPECT_EQ(path, "/log/system_monitor/");
}

TEST_F(TestSystemMonitorBase, GetIsAlarm)
{
    bool alarm = false;
    if (nullptr != instance) {
        alarm = instance->GetIsAlarm();
    }

    EXPECT_EQ(alarm, true);
}

TEST_F(TestSystemMonitorBase, SetIsAlarm)
{
    bool alarm = true;
    if (nullptr != instance) {
        instance->SetIsAlarm(false);
        alarm = instance->GetIsAlarm();
    }

    EXPECT_EQ(alarm, false);
}

TEST_F(TestSystemMonitorBase, GetAlarmValue)
{
    int value = 0;
    if (nullptr != instance) {
        value = instance->GetAlarmValue();
    }

    EXPECT_EQ(value, 60);
}

TEST_F(TestSystemMonitorBase, SetAlarmValue)
{
    int value = 0;
    if (nullptr != instance) {
        instance->SetAlarmValue(10);
        value = instance->GetAlarmValue();
    }

    EXPECT_EQ(value, 10);
}

TEST_F(TestSystemMonitorBase, SetRecordStr)
{
    if (nullptr != instance) {
        instance->SetRecordStr("ut test");
    }
}

TEST_F(TestSystemMonitorBase, StartRecord)
{
    if (nullptr != instance) {
        instance->SetRecordStr("ut test");
        instance->StartRecord();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

TEST_F(TestSystemMonitorBase, StopRecord)
{
    if (nullptr != instance) {
        instance->StopRecord();
    }
}

TEST_F(TestSystemMonitorBase, WriteDataToFile)
{
    if (nullptr != instance) {
        instance->SetRecordStr("ut test");
        instance->WriteDataToFile(true);
    }
}

TEST_F(TestSystemMonitorBase, Notify)
{
    if (nullptr != instance) {
        instance->Notify("");
    }
}

TEST_F(TestSystemMonitorBase, Alarm)
{
    if (nullptr != instance) {
        instance->SetIsAlarm(true);
        instance->Alarm("");
        instance->SetIsAlarm(false);
    }
}

TEST_F(TestSystemMonitorBase, ReportFault)
{
    if (nullptr != instance) {
        instance->ReportFault(405001, 1);
        instance->ReportFault(405001, 0);
    }
}

TEST_F(TestSystemMonitorBase, Control)
{
    if (nullptr != instance) {
        instance->Control(SystemMonitorSubFunctionControlType::kMonitorSwitch, "on");
        instance->Control(SystemMonitorSubFunctionControlType::kMonitorCycle, "3000");
        instance->Control(SystemMonitorSubFunctionControlType::kRecordFileCycle, "10");
        instance->Control(SystemMonitorSubFunctionControlType::kRecordFilePath, "/log/");
        instance->Control(SystemMonitorSubFunctionControlType::kIsAlarm, "1");
        instance->Control(SystemMonitorSubFunctionControlType::kAlarmValue, "60");
        instance->Control(SystemMonitorSubFunctionControlType::kRecordFileCycle, "0");
        instance->Control(SystemMonitorSubFunctionControlType::kMonitorSwitch, "off");
    }
}