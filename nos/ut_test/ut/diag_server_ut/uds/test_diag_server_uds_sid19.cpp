#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid19.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid19 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid19* instancesid19 = nullptr;

TEST_F(TestDiagServerUdsSid19, DiagServerUdsSid19)
{
    instancesid19 = new DiagServerUdsSid19();
}

TEST_F(TestDiagServerUdsSid19, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid19 != nullptr) {
        instancesid19->AnalyzeUdsMessage(udsMessage);
    }
}

TEST_F(TestDiagServerUdsSid19, sendNegative)
{
    DiagServerNrcErrc eNrc = DiagServerNrcErrc::kRequestOutOfRange;
    if (instancesid19 != nullptr) {
        instancesid19->sendNegative(eNrc);
    }
}