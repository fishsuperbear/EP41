#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid34.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid34 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid34* instancesid34 = nullptr;

TEST_F(TestDiagServerUdsSid34, DiagServerUdsSid34)
{
    instancesid34 = new DiagServerUdsSid34();
}

TEST_F(TestDiagServerUdsSid34, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid34 != nullptr) {
        instancesid34->AnalyzeUdsMessage(udsMessage);
    }
}