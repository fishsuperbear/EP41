#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/security/diag_server_security_mgr.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerSecurityMgr : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerSecurityMgr, DiagServerSecurityMgr)
{
}

TEST_F(TestDiagServerSecurityMgr, getInstance)
{
}

TEST_F(TestDiagServerSecurityMgr, DeInit)
{
    // DiagServerSecurityMgr::getInstance()->Init();
    // DiagServerSecurityMgr::getInstance()->DeInit();
}

TEST_F(TestDiagServerSecurityMgr, Init)
{
    // DiagServerSecurityMgr::getInstance()->Init();
}

TEST_F(TestDiagServerSecurityMgr, SessionStatusChange)
{
    DiagServerSessionCode session = DiagServerSessionCode::kDefaultSession;
    DiagServerSecurityMgr::getInstance()->SessionStatusChange(session);
}

TEST_F(TestDiagServerSecurityMgr, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    std::vector<uint8_t> data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x07));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x05));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x06));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x11));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x12));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    data.push_back(static_cast<uint8_t>(0x04));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);

    data.clear();
    data.push_back(static_cast<uint8_t>(0x27));
    data.push_back(static_cast<uint8_t>(0x03));
    udsMessage.udsData = data;
    DiagServerSecurityMgr::getInstance()->AnalyzeUdsMessage(udsMessage);
    // for (int i = 0; i < 58; ++i) {
    //     data.push_back(static_cast<uint8_t>(0x2f));
    // }
}