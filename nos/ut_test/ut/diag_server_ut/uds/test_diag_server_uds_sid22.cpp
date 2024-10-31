#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid22.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid22 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid22* instancesid22 = nullptr;

TEST_F(TestDiagServerUdsSid22, DiagServerUdsSid22)
{
    instancesid22 = new DiagServerUdsSid22();
}

TEST_F(TestDiagServerUdsSid22, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    if (instancesid22 != nullptr) {
        instancesid22->AnalyzeUdsMessage(udsMessage);
    }
}

TEST_F(TestDiagServerUdsSid22, DidSupportAndSecurityCheck)
{
    if (instancesid22 != nullptr) {
        instancesid22->DidSupportAndSecurityCheck(0xf190);
    }
}

TEST_F(TestDiagServerUdsSid22, ReadDidData)
{
    uint16_t did = 0xF18B;
    std::vector<uint8_t> udsData;
    if (instancesid22 != nullptr) {
        instancesid22->ReadDidData(did, udsData);
    }
}

TEST_F(TestDiagServerUdsSid22, RequestToExternalService)
{
    std::vector<std::string> service;
    DiagServerUdsMessage udsMessage;
    if (instancesid22 != nullptr) {
        instancesid22->RequestToExternalService(service, udsMessage);
    }
}