#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid28.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid28 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid28* instancesid28 = nullptr;

TEST_F(TestDiagServerUdsSid28, DiagServerUdsSid28)
{
    instancesid28 = new DiagServerUdsSid28();
}

TEST_F(TestDiagServerUdsSid28, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid28 != nullptr) {
        instancesid28->AnalyzeUdsMessage(udsMessage);
    }
}