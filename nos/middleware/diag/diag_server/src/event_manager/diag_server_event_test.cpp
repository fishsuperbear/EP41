/*!
 * @file diag_server_event_test.cpp
 * This file contains the implementation of the subscriber functions.
 */

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/event_manager/diag_server_event_test.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include <sys/time.h>

namespace hozon {
namespace netaos {
namespace diag {


struct timeval tStart, tEnd;
void StartT()
{
    gettimeofday(&tStart,NULL);
}

void EndT()
{
    gettimeofday(&tEnd,NULL);
    int milsec = (tEnd.tv_sec*1000 + tEnd.tv_usec/1000) - (tStart.tv_sec*1000 + tStart.tv_usec/1000);
    DG_INFO << "DiagServerEventTest cost time:" << milsec;
}

DiagServerEventTest::DiagServerEventTest()
{
    DG_INFO << "DiagServerEventTest::DiagServerEventTest";
}

DiagServerEventTest::~DiagServerEventTest()
{
    DG_INFO << "DiagServerEventTest::~DiagServerEventTest";
}

void DiagServerEventTest::init()
{
    DG_INFO << "DiagServerEventTest::init";
    // test
    m_spTestSub = std::make_shared<testDiagEventPubSubType>();
    m_spTestProxy = std::make_shared<hozon::netaos::cm::Proxy>(m_spTestSub);
    m_spTestProxy->Init(0, "testDiagEvent");
    m_spTestProxy->Listen(std::bind(&DiagServerEventTest::recvTestCallback, this));
    m_testDiagEventData = std::make_shared<testDiagEvent>();
}

void DiagServerEventTest::deInit()
{
    DG_INFO << "DiagServerEventTest::deInit";
    m_spTestProxy->Deinit();
}

void DiagServerEventTest::recvTestCallback()
{
    printf( "DiagServerEventTest::recvTestCallback\n" );
    if (!m_spTestProxy->IsMatched()) {
        return;
    }

    m_spTestProxy->Take(m_testDiagEventData);
    int icmd = m_testDiagEventData->iCmd();
    DG_INFO << "icmd: " << icmd;
    std::vector<uint8_t> param = m_testDiagEventData->data_vec();
    std::string temp;
    for (auto i : param) {
        temp += " " + std::to_string(i);
    }
    DG_INFO << "DiagServerEventTest::recvTestCallback msg:" << temp;

    switch (icmd) {
    case 0x19:
        {
            if (param.size() < 2) break;

            if (param[1] == 0x01) {
                if (param.size() < 2) break;
                DiagServerEventHandler::getInstance()->reportNumberOfDTCByStatusMask(param[2]);
            }
            else if (param[1] == 0x02) {
                if (param.size() < 3) break;
                DiagServerEventHandler::getInstance()->reportDTCByStatusMask(param[2]);
            }
            else if (param[1] == 0x03) {
                DiagServerEventHandler::getInstance()->reportDTCSnapshotIdentification();
            }
            else if (param[1] == 0x04) {
                if (param.size() < 6) break;
                uint8_t dtcHighByte = param[2];
                uint8_t dtcMidByte = param[3];
                uint8_t dtcLowByte = param[4];
                uint32_t dtcValue = dtcHighByte << 16 |  dtcMidByte << 8 | dtcLowByte;
                uint8_t dtcSsrId = param[5];
                DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(dtcValue, dtcSsrId);
            }
            else if (param[1] == 0x0A) {
                DiagServerEventHandler::getInstance()->reportSupportedDTC();
            }
        }
        break;
    case 0x14:
        {
            if (param.size() < 4) break;
            uint8_t groupHighByte = param[1];
            uint8_t groupMidByte = param[2];
            uint8_t groupLowByte = param[3];
            uint32_t dtcGroup = 0x00FFFFFF & (groupHighByte << 16 | groupMidByte << 8 | groupLowByte);
            DiagServerEventHandler::getInstance()->clearDTCInformation(dtcGroup);
        }
        break;
    case 0x85:
        {
            if (param.size() < 2) break;
            uint8_t sw = param[1];
            DIAG_CONTROLDTCSTATUSTYPE controlDtcStatusType = DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOn;
            if (0x02 == sw) {
                controlDtcStatusType = DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOff;
            }

            DiagServerEventHandler::getInstance()->controlDTCStatusType(controlDtcStatusType);
        }
        break;
    case 0x06:
        DiagServerEventHandler::getInstance()->requestOutputDtcInfo();
        break;
    case 0x07:
        {
            if (param.size() < 5) break;
            uint8_t dtcHighByte = param[1];
            uint8_t dtcMidByte = param[2];
            uint8_t dtcLowByte = param[3];
            uint32_t dtcValue = dtcHighByte << 16 |  dtcMidByte << 8 | dtcLowByte;
            uint8_t eventStatus = param[4];
            DiagServerEventHandler::getInstance()->reportDTCEvent(dtcValue, eventStatus);
        }
        break;
    case 0x08:
        {
            StartT();
            uint32_t dtcGroup = 0x00FFFFFF;
            DiagServerEventHandler::getInstance()->clearDTCInformation(dtcGroup);
            EndT();

            StartT();
            uint32_t dtcValue = 0x123456;
            uint8_t eventStatus = 01;
            DiagServerEventHandler::getInstance()->reportDTCEvent(dtcValue, eventStatus);
            EndT();

            StartT();
            uint8_t dtcStatusMask = 0xFF;
            DiagServerEventHandler::getInstance()->reportNumberOfDTCByStatusMask(dtcStatusMask);
            EndT();

            StartT();
            DiagServerEventHandler::getInstance()->reportDTCByStatusMask(dtcStatusMask);
            EndT();

            StartT();
            DiagServerEventHandler::getInstance()->reportDTCSnapshotIdentification();
            EndT();

            StartT();
            uint8_t dtcSsrId = 01;
            DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(dtcValue, dtcSsrId);
            EndT();

            StartT();
            DiagServerEventHandler::getInstance()->reportSupportedDTC();
            EndT();

            StartT();
            DIAG_CONTROLDTCSTATUSTYPE controlDtcStatusType = DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOff;
            DiagServerEventHandler::getInstance()->controlDTCStatusType(controlDtcStatusType);
            EndT();

            StartT();
            DiagServerEventHandler::getInstance()->requestOutputDtcInfo();
            EndT();
        }
    default:
        break;
    }

}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
