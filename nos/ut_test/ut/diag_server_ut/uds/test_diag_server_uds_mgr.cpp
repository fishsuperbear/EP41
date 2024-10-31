#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/uds/diag_server_uds_mgr.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsMgr : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(TestDiagServerUdsMgr, DiagServerUdsMgr)
{
}

TEST_F(TestDiagServerUdsMgr, getInstance)
{
}

TEST_F(TestDiagServerUdsMgr, DeInit)
{
    // DiagServerUdsMgr::getInstance()->Init();
    // DiagServerUdsMgr::getInstance()->DeInit();
}

TEST_F(TestDiagServerUdsMgr, Init)
{
    // DiagServerUdsMgr::getInstance()->Init();
}

TEST_F(TestDiagServerUdsMgr, AnalyzeUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerUdsMgr::getInstance()->AnalyzeUdsMessage(udsMessage);
}

TEST_F(TestDiagServerUdsMgr, sendNegativeResponse)
{
    DiagServerUdsMessage udsMessage;
    DiagServerServiceRequestOpc eOpc = DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER;
    DiagServerUdsMgr::getInstance()->sendNegativeResponse(eOpc, udsMessage);
}

TEST_F(TestDiagServerUdsMgr, sendPositiveResponse)
{
    DiagServerUdsMessage udsMessage;
    DiagServerServiceRequestOpc eOpc = DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER;
    DiagServerUdsMgr::getInstance()->sendPositiveResponse(eOpc, udsMessage);
}

TEST_F(TestDiagServerUdsMgr, getSidService)
{
    DiagServerServiceRequestOpc eOpc = DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER;
    DiagServerUdsMgr::getInstance()->getSidService(eOpc);
}