#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid2F.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid2F : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid2F* instancesid2F = nullptr;

TEST_F(TestDiagServerUdsSid2F, DiagServerUdsSid2F)
{
    instancesid2F = new DiagServerUdsSid2F();
}

TEST_F(TestDiagServerUdsSid2F, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid2F != nullptr) {
        instancesid2F->AnalyzeUdsMessage(udsMessage);
    }
}