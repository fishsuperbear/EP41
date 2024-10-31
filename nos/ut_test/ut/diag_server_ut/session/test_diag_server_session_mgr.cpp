#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/session/diag_server_session_mgr.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerSessionMgr : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerSessionMgr, DiagServerSessionMgr)
{
}

TEST_F(TestDiagServerSessionMgr, getInstance)
{
}

TEST_F(TestDiagServerSessionMgr, DeInit)
{
    DiagServerSessionMgr::getInstance()->Init();
    DiagServerSessionMgr::getInstance()->DeInit();
}

TEST_F(TestDiagServerSessionMgr, Init)
{
    DiagServerSessionMgr::getInstance()->Init();
}

TEST_F(TestDiagServerSessionMgr, DealwithSessionLayerService)
{
    DiagServerUdsMessage udsMessage;
    DiagServerSessionMgr::getInstance()->DealwithSessionLayerService(udsMessage);
}

TEST_F(TestDiagServerSessionMgr, DealwithApplicationLayerService)
{
    DiagServerUdsMessage udsMessage;
    DiagServerSessionMgr::getInstance()->DealwithApplicationLayerService(udsMessage);
}

TEST_F(TestDiagServerSessionMgr, DealwithSpecialSessionRetention)
{
    DiagServerSessionMgr::getInstance()->DealwithSpecialSessionRetention(true);
}

TEST_F(TestDiagServerSessionMgr, DealwithNeklinkStatusChange)
{
    DiagServerSessionMgr::getInstance()->DealwithNeklinkStatusChange();
}