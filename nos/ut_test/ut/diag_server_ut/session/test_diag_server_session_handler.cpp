#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/session/diag_server_session_handler.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerSessionHandler : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerSessionHandler, DiagServerSessionHandler)
{
}

TEST_F(TestDiagServerSessionHandler, getInstance)
{
}

TEST_F(TestDiagServerSessionHandler, DeInit)
{
    // DiagServerSessionHandler::getInstance()->Init();
    // DiagServerSessionHandler::getInstance()->DeInit();
}

TEST_F(TestDiagServerSessionHandler, Init)
{
    // DiagServerSessionHandler::getInstance()->Init();
}

// TEST_F(TestDiagServerSessionHandler, GeneralBehaviourCheck)
// {
//     DiagServerUdsMessage udsMessage;
//     DiagServerSessionHandler::getInstance()->GeneralBehaviourCheck(udsMessage);
// }

// TEST_F(TestDiagServerSessionHandler, GeneralBehaviourCheckWithSubFunction)
// {
//     DiagServerUdsMessage udsMessage;
//     DiagServerSessionHandler::getInstance()->GeneralBehaviourCheckWithSubFunction(udsMessage;);
// }

TEST_F(TestDiagServerSessionHandler, RecvUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerSessionHandler::getInstance()->RecvUdsMessage(udsMessage);
}

TEST_F(TestDiagServerSessionHandler, ReplyUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

TEST_F(TestDiagServerSessionHandler, ReplyNegativeResponse)
{
    DiagServerServiceRequestOpc sid = DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER;
    uint16_t sourceAddr= 0x10c3;
    DiagServerNrcErrc errorCode = DiagServerNrcErrc::kBusyRepeatRequest;
    DiagServerSessionHandler::getInstance()->ReplyNegativeResponse(sid, sourceAddr, errorCode);
}

TEST_F(TestDiagServerSessionHandler, TransmitUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerSessionHandler::getInstance()->TransmitUdsMessage(udsMessage);
}

TEST_F(TestDiagServerSessionHandler, OnDoipNeklinkStatusChange)
{
    DoipNetlinkStatus doipNetlinkStatus = DoipNetlinkStatus::kUp;
    uint16_t address = 0x1062;
    DiagServerSessionHandler::getInstance()->OnDoipNeklinkStatusChange(doipNetlinkStatus, address);
}