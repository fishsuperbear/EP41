#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventHandler : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerEventHandler, DiagServerEventHandler)
{
}

TEST_F(TestDiagServerEventHandler, getInstance)
{
}

// TEST_F(TestDiagServerEventHandler, DeInit)
// {
//     DiagServerEventHandler::getInstance()->Init();
//     DiagServerEventHandler::getInstance()->DeInit();
// }

// TEST_F(TestDiagServerEventHandler, Init)
// {
//     DiagServerEventHandler::getInstance()->Init();
// }

TEST_F(TestDiagServerEventHandler, clearDTCInformation)
{
    uint32_t dtcGroup = 0x666666;
    DiagServerEventHandler::getInstance()->clearDTCInformation(dtcGroup);
}

TEST_F(TestDiagServerEventHandler, reportNumberOfDTCByStatusMask)
{
    uint8_t dtcStatusMask = 0x01;
    DiagServerEventHandler::getInstance()->reportNumberOfDTCByStatusMask(dtcStatusMask);
}

TEST_F(TestDiagServerEventHandler, reportDTCByStatusMask)
{
    uint8_t serverId = 0x00;
    DiagServerEventHandler::getInstance()->reportDTCByStatusMask(serverId);

    // serverId = 0x01;
    // DiagServerEventHandler::getInstance()->reportDTCByStatusMask(serverId);
}

TEST_F(TestDiagServerEventHandler, reportDTCSnapshotIdentification)
{
//     uint8_t serverId = 0x00;
//     uint16_t address = 0x1061;
    DiagServerEventHandler::getInstance()->reportDTCSnapshotIdentification();

    // serverId = 0x01;
    // DiagServerEventHandler::getInstance()->IsUpdateManagerAdress(address, serverId);

    // address = 0x1062;
    // DiagServerEventHandler::getInstance()->IsUpdateManagerAdress(address, serverId);
}

TEST_F(TestDiagServerEventHandler, reportDTCSnapshotRecordByDTCNumber)
{
    uint32_t serverId = 0x00;
    uint32_t address = 0x0e81;
    DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(address, serverId);

    // serverId = 0x01;
    // DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(address, serverId);

    // address = 0x0e80;
    // DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(address, serverId);
}

TEST_F(TestDiagServerEventHandler, reportSupportedDTC)
{
    // uint8_t serverId = 0x00;
    // uint16_t address = 0x0e81;
    DiagServerEventHandler::getInstance()->reportSupportedDTC();

    // serverId = 0x01;
    // DiagServerEventHandler::getInstance()->reportSupportedDTC();

    // address = 0x0e80;
    // DiagServerEventHandler::getInstance()->reportSupportedDTC();
}

TEST_F(TestDiagServerEventHandler, controlDTCStatusType)
{
    DIAG_CONTROLDTCSTATUSTYPE serverId;
    DiagServerEventHandler::getInstance()->controlDTCStatusType(serverId);
}

TEST_F(TestDiagServerEventHandler, reportDTCEvent)
{
    uint32_t dtcValue = 0x666666;
    uint8_t serverId = 0x01;
    DiagServerEventHandler::getInstance()->reportDTCEvent(dtcValue, serverId);
}

TEST_F(TestDiagServerEventHandler, reportSessionChange)
{
    uint32_t service = 0x00;
    DiagServerEventHandler::getInstance()->reportSessionChange(service);
}

TEST_F(TestDiagServerEventHandler, requestOutputDtcInfo)
{
    // uint8_t sid = 0x22;
    // std::vector<std::string> service;
    DiagServerEventHandler::getInstance()->requestOutputDtcInfo();
}

TEST_F(TestDiagServerEventHandler, replyClearAllDtc)
{
    // uint16_t did = 0xf190;
    // std::vector<std::string> service;
    bool bWrite = false;
    DiagServerEventHandler::getInstance()->replyClearAllDtc(bWrite);
}

TEST_F(TestDiagServerEventHandler, replyNumberOfDTCByStatusMask)
{
    uint32_t rid = 0x6140;
    // std::vector<std::string> service;
    DiagServerEventHandler::getInstance()->replyNumberOfDTCByStatusMask(rid);
}

TEST_F(TestDiagServerEventHandler, replyDTCByStatusMask)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventHandler::getInstance()->replyDTCByStatusMask(dtcInfos);
}

TEST_F(TestDiagServerEventHandler, sortDid)
{
    // std::vector<DiagDtcData> dtcInfos;
    // std::map<int, int> mapCount;
    // DiagServerEventHandler::getInstance()->sortDid(dtcInfos, mapCount);
}

TEST_F(TestDiagServerEventHandler, replyDTCSnapshotIdentification)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventHandler::getInstance()->replyDTCSnapshotIdentification(dtcInfos);
}

TEST_F(TestDiagServerEventHandler, replyDTCSnapshotRecordByDTCNumber)
{
    std::vector<DiagDtcData> dtcInfos;
    uint8_t number = 0x01;
    DiagServerEventHandler::getInstance()->replyDTCSnapshotRecordByDTCNumber(dtcInfos, number);
}

TEST_F(TestDiagServerEventHandler, replySupportedDTC)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventHandler::getInstance()->replySupportedDTC(dtcInfos);
}

TEST_F(TestDiagServerEventHandler, replyControlDTCStatusType)
{
    DIAG_CONTROLDTCSTATUSTYPE controlDtcStatusType;
    DiagServerEventHandler::getInstance()->replyControlDTCStatusType(controlDtcStatusType);
}

TEST_F(TestDiagServerEventHandler, replyOutputDtcInfo)
{
    DiagServerEventHandler::getInstance()->replyOutputDtcInfo(true);
}