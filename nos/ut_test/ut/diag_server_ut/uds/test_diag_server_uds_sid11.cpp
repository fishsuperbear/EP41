#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_sid11.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsSid11 : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

DiagServerUdsSid11* instancesid11 = nullptr;

TEST_F(TestDiagServerUdsSid11, DiagServerUdsSid11)
{
    instancesid11 = new DiagServerUdsSid11();
}

TEST_F(TestDiagServerUdsSid11, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    std::vector<uint8_t> data;
    // if (instancesid11 != nullptr) {
    //     instancesid11->AnalyzeUdsMessage(udsMessage);

    //     data.push_back(static_cast<uint8_t>(0x11));
    //     data.push_back(static_cast<uint8_t>(0x03));
    //     data.push_back(static_cast<uint8_t>(0x03));
    //     udsMessage.udsData = data;
    //     instancesid11->AnalyzeUdsMessage(udsMessage);

    //     data.clear();
    //     data.push_back(static_cast<uint8_t>(0x11));
    //     data.push_back(static_cast<uint8_t>(0x01));
    //     udsMessage.udsData = data;
    //     instancesid11->AnalyzeUdsMessage(udsMessage);

    //     data.clear();
    //     data.push_back(static_cast<uint8_t>(0x11));
    //     data.push_back(static_cast<uint8_t>(0x00));
    //     udsMessage.udsData = data;
    //     instancesid11->AnalyzeUdsMessage(udsMessage);
    // }

    DiagServerUdsSid11* p = new DiagServerUdsSid11();
    // p->AnalyzeUdsMessage(udsMessage);

    data.push_back(static_cast<uint8_t>(0x11));
    data.push_back(static_cast<uint8_t>(0x03));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    p->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x11));
    data.push_back(static_cast<uint8_t>(0x01));
    udsMessage.udsData = data;
    p->AnalyzeUdsMessage(udsMessage);

    delete p;
}