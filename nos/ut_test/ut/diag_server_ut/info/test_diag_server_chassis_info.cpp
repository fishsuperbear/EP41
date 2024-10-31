#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/info/diag_server_chassis_info.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerChassisInfo : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerChassisInfo, DiagServerChassisInfo)
{
}

TEST_F(TestDiagServerChassisInfo, getInstance)
{
}

TEST_F(TestDiagServerChassisInfo, DeInit)
{
    DiagServerChassisInfo::getInstance()->Init();
    DiagServerChassisInfo::getInstance()->DeInit();
}

TEST_F(TestDiagServerChassisInfo, Init)
{
    DiagServerChassisInfo::getInstance()->Init();
}