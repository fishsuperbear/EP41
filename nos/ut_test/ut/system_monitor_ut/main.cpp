#include <gtest/gtest.h>
#include "system_monitor/include/common/system_monitor_logger.h"

using namespace hozon::netaos::system_monitor;

int main(int argc, char** argv)
{
    SystemMonitorLogger::GetInstance().InitLogging(
        "system_monitor_logger_ut",
        "system_monitor_logger_ut",
        SystemMonitorLogger::SystemMonitorLogLevelType::SYSTEM_MONITOR_OFF);
    SystemMonitorLogger::GetInstance().CreateLogger("system_monitor_logger_ut");

    ::testing::InitGoogleTest(&argc, argv);
    // ::testing::InitGoogleMock(&argc, argv);
    // ::testing::GTEST_FLAG(color) = "yes";
    int ret = RUN_ALL_TESTS();
    return ret;
}