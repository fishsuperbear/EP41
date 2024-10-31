#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid14.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid14 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid14* instancesid14 = nullptr;

TEST_F(TestDiagServerUdsSid14, DiagServerUdsSid14)
{
    instancesid14 = new DiagServerUdsSid14();
}

TEST_F(TestDiagServerUdsSid14, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    std::vector<uint8_t> data;
    if (instancesid14 != nullptr) {

        instancesid14->AnalyzeUdsMessage(udsMessage);

        data.push_back(static_cast<uint8_t>(0x14));
        data.push_back(static_cast<uint8_t>(0xff));
        data.push_back(static_cast<uint8_t>(0xff));
        data.push_back(static_cast<uint8_t>(0xff));
        udsMessage.udsData = data;
        instancesid14->AnalyzeUdsMessage(udsMessage);
    }
}