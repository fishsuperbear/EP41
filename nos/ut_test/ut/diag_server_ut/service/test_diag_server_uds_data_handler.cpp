#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerUdsDataHandler : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerUdsDataHandler, DiagServerUdsDataHandler)
{
}

TEST_F(TestDiagServerUdsDataHandler, getInstance)
{
}

TEST_F(TestDiagServerUdsDataHandler, DeInit)
{
    DiagServerUdsDataHandler::getInstance()->Init();
    DiagServerUdsDataHandler::getInstance()->DeInit();
}

TEST_F(TestDiagServerUdsDataHandler, Init)
{
    DiagServerUdsDataHandler::getInstance()->Init();
}

TEST_F(TestDiagServerUdsDataHandler, RecvUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerUdsDataHandler::getInstance()->RecvUdsMessage(udsMessage);
}

TEST_F(TestDiagServerUdsDataHandler, ReplyUdsMessage)
{
    DiagServerUdsMessage udsMessage;
    DiagServerUdsDataHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

// TEST_F(TestDiagServerUdsDataHandler, GetDiagServerPhysicAddress)
// {
//     uint8_t serverId = 0x00;
//     DiagServerUdsDataHandler::getInstance()->GetDiagServerPhysicAddress(serverId);

//     serverId = 0x01;
//     DiagServerUdsDataHandler::getInstance()->GetDiagServerPhysicAddress(serverId);
// }

// TEST_F(TestDiagServerUdsDataHandler, IsUpdateManagerAdress)
// {
//     uint8_t serverId = 0x00;
//     uint16_t address = 0x1061;
//     DiagServerUdsDataHandler::getInstance()->IsUpdateManagerAdress(address, serverId);

//     serverId = 0x01;
//     DiagServerUdsDataHandler::getInstance()->IsUpdateManagerAdress(address, serverId);

//     address = 0x1062;
//     DiagServerUdsDataHandler::getInstance()->IsUpdateManagerAdress(address, serverId);
// }