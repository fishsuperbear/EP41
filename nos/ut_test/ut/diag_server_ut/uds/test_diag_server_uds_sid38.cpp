#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid38.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid38 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid38* instancesid38 = nullptr;

TEST_F(TestDiagServerUdsSid38, DiagServerUdsSid38)
{
    instancesid38 = new DiagServerUdsSid38();
}

TEST_F(TestDiagServerUdsSid38, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid38 != nullptr) {
        instancesid38->AnalyzeUdsMessage(udsMessage);
    }
}