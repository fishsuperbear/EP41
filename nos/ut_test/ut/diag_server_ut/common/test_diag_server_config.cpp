// This is googletest template, please change TestCaseName and TestName, write you own test code here.

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "diag/diag_server/include/common/diag_server_config.h"
// #define private public
// #define protected public

using namespace hozon::netaos::diag;

class TestDiagServerConfig : public testing::Test
{
    friend class DiagServerConfig;
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(TestDiagServerConfig, DiagServerConfig)
{
}

TEST_F(TestDiagServerConfig, getInstance)
{
}

TEST_F(TestDiagServerConfig, DeInit)
{
    DiagServerConfig::getInstance()->Init();
    DiagServerConfig::getInstance()->DeInit();
}

TEST_F(TestDiagServerConfig, Init)
{
    DiagServerConfig::getInstance()->Init();
}

TEST_F(TestDiagServerConfig, LoadDiagConfig)
{
    DiagServerConfig::getInstance()->LoadDiagConfig();
}

TEST_F(TestDiagServerConfig, LoadDMAndDTCConfig)
{
    // DiagServerConfig::getInstance()->LoadDMAndDTCConfig();
}

TEST_F(TestDiagServerConfig, GetDiagServerPhysicAddress)
{
    uint8_t serverId = 0x00;
    DiagServerConfig::getInstance()->GetDiagServerPhysicAddress(serverId);

    serverId = 0x01;
    DiagServerConfig::getInstance()->GetDiagServerPhysicAddress(serverId);
}

TEST_F(TestDiagServerConfig, IsUpdateManagerAdress)
{
    uint8_t serverId = 0x00;
    uint16_t address = 0x1061;
    DiagServerConfig::getInstance()->IsUpdateManagerAdress(address, serverId);

    serverId = 0x01;
    DiagServerConfig::getInstance()->IsUpdateManagerAdress(address, serverId);

    address = 0x1062;
    DiagServerConfig::getInstance()->IsUpdateManagerAdress(address, serverId);
}

TEST_F(TestDiagServerConfig, IsRemoteAdress)
{
    uint8_t serverId = 0x00;
    uint16_t address = 0x0e81;
    DiagServerConfig::getInstance()->IsRemoteAdress(address, serverId);

    serverId = 0x01;
    DiagServerConfig::getInstance()->IsRemoteAdress(address, serverId);

    address = 0x0e80;
    DiagServerConfig::getInstance()->IsRemoteAdress(address, serverId);
}

TEST_F(TestDiagServerConfig, IsUpdateManagerOrRemoteAdress)
{
    uint8_t serverId = 0x00;
    uint16_t address = 0x0e81;
    DiagServerConfig::getInstance()->IsUpdateManagerOrRemoteAdress(address, serverId);

    serverId = 0x01;
    DiagServerConfig::getInstance()->IsUpdateManagerOrRemoteAdress(address, serverId);

    address = 0x0e80;
    DiagServerConfig::getInstance()->IsUpdateManagerOrRemoteAdress(address, serverId);
}

TEST_F(TestDiagServerConfig, IsSupportDoip)
{
    uint8_t serverId = 0x01;
    DiagServerConfig::getInstance()->IsSupportDoip(serverId);
}

TEST_F(TestDiagServerConfig, IsSupportDoCan)
{
    uint8_t serverId = 0x01;
    DiagServerConfig::getInstance()->IsSupportDoCan(serverId);
}

TEST_F(TestDiagServerConfig, IsSupportDoSomeip)
{
    uint8_t serverId = 0x01;
    DiagServerConfig::getInstance()->IsSupportDoSomeip(serverId);
}

TEST_F(TestDiagServerConfig, QueryAllExternalService)
{
    std::vector<std::string> service;
    DiagServerConfig::getInstance()->QueryAllExternalService(service);
}

TEST_F(TestDiagServerConfig, QueryExternalServiceBySid)
{
    uint8_t sid = 0x22;
    std::vector<std::string> service;
    DiagServerConfig::getInstance()->QueryExternalServiceBySid(sid, service);
}

TEST_F(TestDiagServerConfig, QueryExternalServiceByDid)
{
    uint16_t did = 0xf190;
    std::vector<std::string> service;
    bool bWrite = false;
    DiagServerConfig::getInstance()->QueryExternalServiceByDid(did, service, bWrite);
}

TEST_F(TestDiagServerConfig, QueryExternalServiceByRid)
{
    uint16_t rid = 0x6140;
    std::vector<std::string> service;
    DiagServerConfig::getInstance()->QueryExternalServiceByRid(rid, service);
}

TEST_F(TestDiagServerConfig, GetDiagServerDataTranferSize)
{
    uint8_t configId = 0x00;
    DiagServerConfig::getInstance()->GetDiagServerDataTranferSize(configId);
}

TEST_F(TestDiagServerConfig, QueryDataTranferConfig)
{
    uint8_t configId = 0x00;
    DiagTransferConfigDataInfo configInfo;
    DiagServerConfig::getInstance()->QueryDataTranferConfig(configId, configInfo);
}

TEST_F(TestDiagServerConfig, GetDiagServerSecurityMask)
{
    uint8_t levelId = 0x00;
    DiagServerConfig::getInstance()->GetDiagServerSecurityMask(levelId);
}

TEST_F(TestDiagServerConfig, QueryAccessPermissionBySid)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryAccessPermissionBySid(sid, accessInfo);
}

TEST_F(TestDiagServerConfig, QueryAccessPermissionBySidAndSubFunc)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    uint8_t subFuc = 0x00;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryAccessPermissionBySidAndSubFunc(sid, subFuc, accessInfo);
}

TEST_F(TestDiagServerConfig, QueryReadAccessPermissionByDid)
{
    uint16_t did = 0xf188;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryReadAccessPermissionByDid(did, accessInfo);
}

TEST_F(TestDiagServerConfig, QueryWriteAccessPermissionByDid)
{
    uint16_t did = 0xf188;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryWriteAccessPermissionByDid(did, accessInfo);
}

TEST_F(TestDiagServerConfig, QueryAccessPermissionByRid)
{
    uint16_t rid = 0x6140;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryAccessPermissionByRid(rid, accessInfo);
}

TEST_F(TestDiagServerConfig, QuerySecurityLevelInfoByName)
{
    uint8_t levelId = 0x00;
    DiagSecurityLevelDataInfo levelInfo;
    DiagServerConfig::getInstance()->QuerySecurityLevelInfoByName(levelId, levelInfo);
}

TEST_F(TestDiagServerConfig, QueryAccessPermissionByName)
{
    uint8_t acceccId = 0x00;
    DiagAccessPermissionDataInfo accessInfo;
    DiagServerConfig::getInstance()->QueryAccessPermissionByName(acceccId, accessInfo);
}

TEST_F(TestDiagServerConfig, GetSidMaxPendingNum)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    DiagServerConfig::getInstance()->GetSidMaxPendingNum(sid);
}

TEST_F(TestDiagServerConfig, QuerySidSupport)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    DiagServerConfig::getInstance()->QuerySidSupport(sid);
}

TEST_F(TestDiagServerConfig, QuerySidSupportInActiveSession)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    uint16_t sourceAddr = 0x00;
    DiagServerConfig::getInstance()->QuerySidSupportInActiveSession(sid, sourceAddr);
}

TEST_F(TestDiagServerConfig, QuerySubFunctionSupportInActiveSession)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    uint8_t subFunc = 0x00;
    DiagServerConfig::getInstance()->QuerySubFunctionSupportInActiveSession(sid, subFunc);
}

TEST_F(TestDiagServerConfig, QuerySubFunctionSupportForSid)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    uint8_t subFunc = 0x00;
    DiagServerConfig::getInstance()->QuerySubFunctionSupportForSid(sid, subFunc);
}

TEST_F(TestDiagServerConfig, QuerySidHaveSubFunction)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    DiagServerConfig::getInstance()->QuerySidHaveSubFunction(sid);
}

TEST_F(TestDiagServerConfig, QuerySubFunctionSupportSuppressPosMsgindication)
{
    DiagServerServiceRequestOpc sid = DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR;
    uint8_t subFunc = 0x00;
    DiagServerConfig::getInstance()->QuerySubFunctionSupportSuppressPosMsgindication(sid, subFunc);
}

TEST_F(TestDiagServerConfig, QueryDidDataSize)
{
    uint16_t did = 0xf188;
    DiagServerConfig::getInstance()->QueryDidDataSize(did);
}

TEST_F(TestDiagServerConfig, QueryRidSupport)
{
    uint16_t rid = 0x6140;
    DiagServerConfig::getInstance()->QueryRidSupport(rid);
}

TEST_F(TestDiagServerConfig, QueryRidMultiStartSupport)
{
    uint16_t rid = 0x6140;
    DiagServerConfig::getInstance()->QueryRidMultiStartSupport(rid);
}

TEST_F(TestDiagServerConfig, QueryRidSupportList)
{
    std::vector<uint16_t> ridList;
    DiagServerConfig::getInstance()->QueryRidSupportList(ridList);
}

TEST_F(TestDiagServerConfig, QueryRidDataInfo)
{
    uint16_t rid = 0x6140;
    DiagRidDataInfo dataInfo;
    DiagServerConfig::getInstance()->QueryRidDataInfo(rid, dataInfo);
}

TEST_F(TestDiagServerConfig, QueryRidDataLenBySubFunction)
{
    uint16_t rid = 0x6140;
    uint8_t subFunc = 0x00;
    size_t length = 0x10;
    bool isReply = false;
    DiagServerConfig::getInstance()->QueryRidDataLenBySubFunction(rid, subFunc, length, isReply);
}

TEST_F(TestDiagServerConfig, QuerySessionP2Timer)
{
    DiagServerSessionCode session;
    DiagServerConfig::getInstance()->QuerySessionP2Timer(session);
}

TEST_F(TestDiagServerConfig, QuerySessionP2StarTimer)
{
    DiagServerSessionCode session;
    DiagServerConfig::getInstance()->QuerySessionP2StarTimer(session);
}

TEST_F(TestDiagServerConfig, QuerySessionS3Timer)
{
    DiagServerSessionCode session;
    DiagServerConfig::getInstance()->QuerySessionS3Timer(session);
}

TEST_F(TestDiagServerConfig, GetAllDtc)
{
    std::vector<uint32_t> allDtc;
    DiagServerConfig::getInstance()->GetAllDtc(allDtc);
}

TEST_F(TestDiagServerConfig, GetSingleDtcData)
{
    uint32_t dtc = 0x652181;
    DiagDtcDataInfo outDtcData;
    DiagServerConfig::getInstance()->GetSingleDtcData(dtc, outDtcData);
}

TEST_F(TestDiagServerConfig, CheckDtcIsExsit)
{
    uint32_t dtc = 0x652181;
    DiagServerConfig::getInstance()->CheckDtcIsExsit(dtc);
}

TEST_F(TestDiagServerConfig, GetDemInfo)
{
    DiagDemDataInfo demInfo;
    DiagServerConfig::getInstance()->GetDemInfo(demInfo);
}

TEST_F(TestDiagServerConfig, QueryPrintConfigData)
{
    DiagServerConfig::getInstance()->QueryPrintConfigData();
}