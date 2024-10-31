#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid37.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid37 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid37* instancesid37 = nullptr;

TEST_F(TestDiagServerUdsSid37, DiagServerUdsSid37)
{
    instancesid37 = new DiagServerUdsSid37();
}

TEST_F(TestDiagServerUdsSid37, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid37 != nullptr) {
        instancesid37->AnalyzeUdsMessage(udsMessage);
    }
}