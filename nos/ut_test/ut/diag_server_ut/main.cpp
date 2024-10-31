/*
 * main.cpp
 */
#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/common/diag_server_logger.h"

using namespace hozon::netaos::diag;
void InitLog()
{
    DiagServerLogger::GetInstance().InitLogging("diag_ut", "diag_ut", DiagServerLogger::DiagLogLevelType::DIAG_ERROR);
    // DiagServerLogger::GetInstance().InitLogging("diag_ut");

    // DiagServerLogger::GetInstance().InitLogging("diag_ut", "diag_ut", DiagServerLogger::DiagLogLevelType::DIAG_CRITICAL);
    DiagServerLogger::GetInstance().CreateLogger("diag_ut");
}

int main(int argc, char** argv)
{
    InitLog();
    ::testing::InitGoogleTest(&argc, argv);
    // ::testing::InitGoogleMock(&argc, argv);
    // ::testing::GTEST_FLAG(color) = "yes";
    int ret = RUN_ALL_TESTS();
    return ret;
}
/* EOF */
