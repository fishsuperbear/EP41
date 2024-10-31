#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid36.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid36 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid36* instancesid36 = nullptr;

TEST_F(TestDiagServerUdsSid36, DiagServerUdsSid36)
{
    instancesid36 = new DiagServerUdsSid36();
}

TEST_F(TestDiagServerUdsSid36, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid36 != nullptr) {
        instancesid36->AnalyzeUdsMessage(udsMessage);
    }
}