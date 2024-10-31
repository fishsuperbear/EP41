#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid2E.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid2E : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid2E* instancesid2E = nullptr;

TEST_F(TestDiagServerUdsSid2E, DiagServerUdsSid2E)
{
    instancesid2E = new DiagServerUdsSid2E();
}

TEST_F(TestDiagServerUdsSid2E, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid2E != nullptr) {
        instancesid2E->AnalyzeUdsMessage(udsMessage);
    }
}