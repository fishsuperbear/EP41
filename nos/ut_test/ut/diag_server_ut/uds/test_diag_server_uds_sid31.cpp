#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid31.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid31 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid31* instancesid31 = nullptr;

TEST_F(TestDiagServerUdsSid31, DiagServerUdsSid31)
{
    instancesid31 = new DiagServerUdsSid31();
}

TEST_F(TestDiagServerUdsSid31, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid31 != nullptr) {
        instancesid31->AnalyzeUdsMessage(udsMessage);
    }
}