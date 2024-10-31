#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/event_manager/diag_server_event_status.h"
#include "diag/diag_server/include/event_manager/diag_server_event_impl.h"

// #define private public
// #define protected public
using namespace hozon::netaos::diag;

class TestDiagServerEventImpl : public testing::Test
{
   protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(TestDiagServerEventImpl, DiagServerEventImpl)
{
}

TEST_F(TestDiagServerEventImpl, getInstance)
{
}

TEST_F(TestDiagServerEventImpl, destroy)
{
    // DiagServerEventImpl::getInstance()->Init();
    // DiagServerEventImpl::getInstance()->destroy();
}

TEST_F(TestDiagServerEventImpl, fileExists)
{
    DiagServerEventImpl::getInstance()->fileExists("/app/version.json");
}

TEST_F(TestDiagServerEventImpl, newCircle)
{
    DiagServerEventImpl::getInstance()->newCircle();
}

TEST_F(TestDiagServerEventImpl, getAgingDtcs)
{
    std::unordered_map<uint32_t, uint32_t> outAgingDtcs;
    DiagServerEventImpl::getInstance()->getAgingDtcs(outAgingDtcs);

    // serverId = 0x01;
    // DiagServerEventImpl::getInstance()->getAgingDtcs(serverId);
}

TEST_F(TestDiagServerEventImpl, checkDbFileAndCreateTable)
{
    // uint8_t serverId = 0x00;
    // uint16_t address = 0x1061;
    DiagServerEventImpl::getInstance()->checkDbFileAndCreateTable();

    // serverId = 0x01;
    // DiagServerEventImpl::getInstance()->IsUpdateManagerAdress(address, serverId);

    // address = 0x1062;
    // DiagServerEventImpl::getInstance()->IsUpdateManagerAdress(address, serverId);
}

TEST_F(TestDiagServerEventImpl, createFileAndTables)
{
    // uint8_t serverId = 0x00;
    // uint16_t address = 0x0e81;
    DiagServerEventImpl::getInstance()->createFileAndTables();
}

TEST_F(TestDiagServerEventImpl, sqliteOperator)
{
    // uint8_t serverId = 0x00;
    // uint16_t address = 0x0e81;
    DiagServerEventImpl::getInstance()->sqliteOperator("/app/version.json", "serverId");

    // serverId = 0x01;
    // DiagServerEventImpl::getInstance()->sqliteOperator(address, serverId);

    // address = 0x0e80;
    // DiagServerEventImpl::getInstance()->sqliteOperator(address, serverId);
}

TEST_F(TestDiagServerEventImpl, clearDTCInformation)
{
    uint32_t serverId = 0x01;
    DiagServerEventImpl::getInstance()->clearDTCInformation(serverId);
}

TEST_F(TestDiagServerEventImpl, getAllDtc)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->getAllDtc(dtcInfos);
}

TEST_F(TestDiagServerEventImpl, getGroupDtc)
{
    std::string group;
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->getGroupDtc(group, dtcInfos);
}

TEST_F(TestDiagServerEventImpl, deleteAllDtc)
{
    // std::vector<std::string> service;
    DiagServerEventImpl::getInstance()->deleteAllDtc();
}

TEST_F(TestDiagServerEventImpl, deleteGroupDtc)
{
    std::string service;
    DiagServerEventImpl::getInstance()->deleteGroupDtc(service);
}

TEST_F(TestDiagServerEventImpl, reportDTCByStatusMask)
{
    uint8_t dtcStatusMask = 0x01;
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCByStatusMask(dtcStatusMask, dtcInfos);
}

TEST_F(TestDiagServerEventImpl, reportDTCSnapshotIdentification)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCSnapshotIdentification(dtcInfos);
}

TEST_F(TestDiagServerEventImpl, reportDTCSnapshotRecordByDTCNumber)
{
    uint32_t dtc = 0x666666;
    uint16_t ssrNumber = 0x01;
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportDTCSnapshotRecordByDTCNumber(dtc, ssrNumber, dtcInfos);
}

TEST_F(TestDiagServerEventImpl, reportSupportedDTC)
{
    std::vector<DiagDtcData> dtcInfos;
    DiagServerEventImpl::getInstance()->reportSupportedDTC(dtcInfos);
}

TEST_F(TestDiagServerEventImpl, reportDTCEvent)
{
    uint32_t dtcValue = 0x666666;
    uint8_t eventStatus = 0x01;
    DiagServerEventImpl::getInstance()->reportDTCEvent(dtcValue, eventStatus);
}

TEST_F(TestDiagServerEventImpl, checkDbEmpty)
{
    DiagServerEventImpl::getInstance()->checkDbEmpty();
}

TEST_F(TestDiagServerEventImpl, dealWithDtcRecover)
{
    // DiagServerEventStatus cDtcStatus;
    // char* p;
    // DiagServerEventImpl::getInstance()->dealWithDtcRecover(cDtcStatus, p);
}

TEST_F(TestDiagServerEventImpl, getInsertDtcSql)
{
    // uint32_t dtcValue = 0x666666;
    // uint8_t iDtcstatus = 0x01;
    // uint8_t iTripcount = 0x02;
    // char* p;
    // DiagServerEventImpl::getInstance()->getInsertDtcSql(dtcValue, iDtcstatus, iTripcount, p);
}

TEST_F(TestDiagServerEventImpl, fillSsrData)
{
    DiagDtcData dtcdata;
    DiagServerEventImpl::getInstance()->fillSsrData(dtcdata);
}

TEST_F(TestDiagServerEventImpl, requestOutputDtcInfo)
{
    // uint16_t rid = 0x6140;
    // DiagAccessPermissionDataInfo accessInfo;
    DiagServerEventImpl::getInstance()->requestOutputDtcInfo();
}

TEST_F(TestDiagServerEventImpl, notifyDtcControlSetting)
{
    DIAG_CONTROLDTCSTATUSTYPE controlDtcStatusType;
    DiagServerEventImpl::getInstance()->notifyDtcControlSetting(controlDtcStatusType);
}