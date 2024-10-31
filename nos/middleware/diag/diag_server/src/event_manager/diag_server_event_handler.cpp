/*
* Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
* Description: diag event handler
*/

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/uds/diag_server_uds_mgr.h"
#include "diag/diag_server/include/event_manager/diag_server_event_task.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include <algorithm>
#include <map>

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerEventHandler::m_lck;
DiagServerEventHandler* DiagServerEventHandler::m_instance = nullptr;
const int DIAG_SERVER_EVENT_THREADPOOL_NUM = 1;
const uint8_t DTC_STATUS_AVAILABILITY_MASK = 0x89;          // dtc status availability mask
const uint8_t SAE_J2012_DA_DTCFORMAT_00 = 0x00;             // SAE_J2012-DA dtc format

enum REPOT_TYPE : uint8_t
{
    REPOT_TYPE_REPORTNUMBEROFDTCBYSTATUSMASK = 0x01,        // reportNumberOfDTCByStatusMask
    REPOT_TYPE_REPORTDTCBYSTATUSMASK = 0x02,                // reportDTCByStatusMask
    REPOT_TYPE_REPORTDTCSNAPSHOTIDENTIFICATION = 0x03,      // reportDTCSnapshotIdentification
    REPOT_TYPE_REPORTDTCSNAPSHOTRECORDBYDTCNUMBER = 0x04,   // reportDTCSnapshotRecordByDTCNumber
    REPOT_TYPE_REPORTSUPPORTEDDTC = 0x0A,                   // reportSupportedDTC
};

DiagServerEventHandler*
DiagServerEventHandler::getInstance()
{
    if (nullptr == m_instance) {
        std::unique_lock<std::mutex> lck(m_lck);
        if (nullptr == m_instance) {
            m_instance = new DiagServerEventHandler();
        }
    }

    return m_instance;
}

DiagServerEventHandler::DiagServerEventHandler()
{
    m_upThreadPool.reset(new ThreadPool(DIAG_SERVER_EVENT_THREADPOOL_NUM));
}

DiagServerEventHandler::~DiagServerEventHandler()
{

}

void
DiagServerEventHandler::clearDTCInformation(const uint32_t dtcGroup)
{
    DG_INFO << "DiagServerEventHandler::clearDTCInformation dtcGroup:" << dtcGroup;
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerClearDTCInfo(dtcGroup));
    return;
}

void
DiagServerEventHandler::reportNumberOfDTCByStatusMask(const uint8_t dtcStatusMask)
{
    DG_INFO << "DiagServerEventHandler::reportNumberOfDTCByStatusMask dtcStatusMask:" << dtcStatusMask;
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportDTCNumByStatusMask(dtcStatusMask));
    return;
}

void
DiagServerEventHandler::reportDTCByStatusMask(const uint8_t dtcStatusMask)
{
    DG_INFO << "DiagServerEventHandler::reportDTCByStatusMask dtcStatusMask:" << dtcStatusMask;
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportDTCByStatusMask(dtcStatusMask));
    return;
}

void
DiagServerEventHandler::reportDTCSnapshotIdentification()
{
    DG_INFO << "DiagServerEventHandler::reportDTCSnapshotIdentification";
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportDTCSnapshotIdentification());
    return;
}

void
DiagServerEventHandler::reportDTCSnapshotRecordByDTCNumber(const uint32_t dtcValue, const uint32_t number)
{
    char format[64]{0};
    snprintf(format, 64, "dtc:0x%06X,number:0x%02X", dtcValue, number);
    DG_INFO << "DiagServerEventHandler::reportDTCSnapshotRecordByDTCNumber " << std::string(format);
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportDTCSnapshotRecordByDTCNumber(dtcValue, number));
    return;
}

void
DiagServerEventHandler::reportSupportedDTC()
{
    DG_INFO << "DiagServerEventHandler::reportSupportedDTC";
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportSupportedDTC());
    return;
}

void
DiagServerEventHandler::controlDTCStatusType(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType)
{
    // DG_INFO << "DiagServerEventHandler::controlDTCStatusType controlDtcStatusType:" << controlDtcStatusType;
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerControlDTCStatusType(controlDtcStatusType));
    return;
}

void
DiagServerEventHandler::reportDTCEvent(uint32_t dtcValue, uint8_t dtcStatus)
{
    char format[64]{0};
    snprintf(format, 64, "faultKey:%d, faultStatus:%d", dtcValue, dtcStatus);
    DG_INFO << "DiagServerEventHandler::reportDTCEvent " << std::string(format);
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerReportDTCEvent(dtcValue, dtcStatus));
    return;
}

void
DiagServerEventHandler::reportSessionChange(uint32_t sessionType)
{
    DG_INFO << "DiagServerEventHandler::reportSessionChange sessionType:" << sessionType;
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerSessionChange(sessionType));
    return;
}

void
DiagServerEventHandler::requestOutputDtcInfo()
{
    DG_INFO << "DiagServerEventHandler::requestOutputDtcInfo";
    if (m_upThreadPool) m_upThreadPool->AddTask(new DiagServerRequestOutputDtcInfo());
    return;
}


void
DiagServerEventHandler::replyClearAllDtc(const bool result)
{
    DG_INFO << "DiagServerEventHandler::replyClearAllDtc";
    if (!result) {
        DiagServerUdsMessage udsNegativeMsg;
        udsNegativeMsg.udsData.push_back(0x7F);
        udsNegativeMsg.udsData.push_back(0x14);
        udsNegativeMsg.udsData.push_back(0x31);
        DiagServerUdsMgr::getInstance()->sendNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR, udsNegativeMsg);
        return;
    }

    DiagServerUdsMessage udsMessage;
    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR, udsMessage);
    return;
}

void
DiagServerEventHandler::replyNumberOfDTCByStatusMask(const uint32_t number)
{
    DG_INFO << "DiagServerEventHandler::replyNumberOfDTCByStatusMask number:" << number;
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO);
    udsMessage.udsData.push_back(REPOT_TYPE_REPORTNUMBEROFDTCBYSTATUSMASK);
    udsMessage.udsData.push_back(DTC_STATUS_AVAILABILITY_MASK);
    udsMessage.udsData.push_back(SAE_J2012_DA_DTCFORMAT_00);

    uint8_t countHighByte = 0xFF & (number >> 8);
    uint8_t countLowByte = 0xFF & number;
    udsMessage.udsData.push_back(countHighByte);
    udsMessage.udsData.push_back(countLowByte);

    std::string temp;
    for (auto i : udsMessage.udsData) {
        char format[8]{0};
        snprintf(format, 8, "0x%02x", i);
        temp += " " + std::string(format);
    }
    DG_INFO << "DiagServerEventHandler::replyNumberOfDTCByStatusMask msg:" << temp;

    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsMessage);
    return;
}

void
DiagServerEventHandler::replyDTCByStatusMask(std::vector<DiagDtcData>& dtcInfos)
{
    DG_INFO << "DiagServerEventHandler::replyDTCByStatusMask";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO);
    udsMessage.udsData.push_back(REPOT_TYPE_REPORTDTCBYSTATUSMASK);
    udsMessage.udsData.push_back(DTC_STATUS_AVAILABILITY_MASK);

    for (auto single : dtcInfos) {
        uint8_t dtcHighByte = single.dtc >> 16;
        uint8_t dtcMiddleByte = single.dtc >> 8;
        uint8_t dtcLowByte = single.dtc;
        uint8_t dtcStatus = single.dtcStatus;
        udsMessage.udsData.push_back(dtcHighByte);
        udsMessage.udsData.push_back(dtcMiddleByte);
        udsMessage.udsData.push_back(dtcLowByte);
        udsMessage.udsData.push_back(dtcStatus);
    }

    std::string temp;
    for (auto i : udsMessage.udsData) {
        char format[8]{0};
        snprintf(format, 8, "0x%02x", i);
        temp += " " + std::string(format);
    }
    DG_INFO << "DiagServerEventHandler::replyDTCByStatusMask msg:" << temp;


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsMessage);
    return;
}

 void sortDid(std::vector<DiagDtcData>& dtcInfos, std::map<int, int>& mapCount)
 {
    if (dtcInfos.empty()) {
        return;
    }

    std::sort(dtcInfos.begin(), dtcInfos.end(),
        [](const DiagDtcData& first, const DiagDtcData& second) { return first.dtc < second.dtc; });

    for (auto& dtcInfo : dtcInfos) {
        mapCount[dtcInfo.dtc]++;
    }

    return;
 }


void
DiagServerEventHandler::replyDTCSnapshotIdentification(std::vector<DiagDtcData>& dtcInfos)
{
    DG_INFO << "DiagServerEventHandler::replyDTCSnapshotIdentification";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO);
    udsMessage.udsData.push_back(REPOT_TYPE_REPORTDTCSNAPSHOTIDENTIFICATION);

    for (auto single : dtcInfos) {
        uint8_t dtcHighByte = single.dtc >> 16;
        uint8_t dtcMiddleByte = single.dtc >> 8;
        uint8_t dtcLowByte = single.dtc;
        uint8_t snapshotId = 0x01;
        udsMessage.udsData.push_back(dtcHighByte);
        udsMessage.udsData.push_back(dtcMiddleByte);
        udsMessage.udsData.push_back(dtcLowByte);
        udsMessage.udsData.push_back(snapshotId);
    }

    std::string temp;
    for (auto i : udsMessage.udsData) {
        char format[8]{0};
        snprintf(format, 8, "0x%02x", i);
        temp += " " + std::string(format);
    }
    DG_INFO << "DiagServerEventHandler::replyDTCSnapshotIdentification msg:" << temp;


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsMessage);
    return;
}

void
DiagServerEventHandler::replyDTCSnapshotRecordByDTCNumber(std::vector<DiagDtcData>& dtcInfos, uint8_t number)
{
    DG_INFO << "DiagServerEventHandler::replyDTCSnapshotRecordByDTCNumber";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO);
    udsMessage.udsData.push_back(REPOT_TYPE_REPORTDTCSNAPSHOTRECORDBYDTCNUMBER);
    if (dtcInfos.empty()) {
        // dtc not exsit config
        DiagServerUdsMessage udsNegativeMsg;
        udsNegativeMsg.udsData.push_back(0x7F);
        udsNegativeMsg.udsData.push_back(0x19);
        udsNegativeMsg.udsData.push_back(0x31);
        DiagServerUdsMgr::getInstance()->sendNegativeResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsNegativeMsg);
        return;
    }

    for (int index = 0; index < (int)dtcInfos.size(); ++index) {
        uint8_t dtcHighByte = dtcInfos[index].dtc >> 16;
        uint8_t dtcMiddleByte = dtcInfos[index].dtc>> 8;
        uint8_t dtcLowByte = dtcInfos[index].dtc;
        uint8_t dtcStatus = dtcInfos[index].dtcStatus;
        udsMessage.udsData.push_back(dtcHighByte);
        udsMessage.udsData.push_back(dtcMiddleByte);
        udsMessage.udsData.push_back(dtcLowByte);
        udsMessage.udsData.push_back(dtcStatus);
        if (dtcStatus) {
            udsMessage.udsData.push_back(number);
        }

        uint8_t didSize = dtcInfos[index].vSnapshotData.size() + dtcInfos[index].vExtendData.size();
        if (didSize > 0) {
            udsMessage.udsData.push_back(didSize);
        }

        // add snapshot data
        for (auto data : dtcInfos[index].vSnapshotData) {
            udsMessage.udsData.insert(udsMessage.udsData.end(), data.didData.begin(), data.didData.end());
        }

        // add extend date
        if (dtcInfos[index].vExtendData.empty()) {
            continue;
        }
        for (auto extendData : dtcInfos[index].vExtendData) {
            udsMessage.udsData.insert(udsMessage.udsData.end(), extendData.didData.begin(), extendData.didData.end());
        }
    }

    std::string temp;
    for (auto i : udsMessage.udsData) {
        char format[8]{0};
        snprintf(format, 8, "0x%02x", i);
        temp += " " + std::string(format);
    }
    DG_INFO << "DiagServerEventHandler::replyDTCSnapshotRecordByDTCNumber msg:" << temp;


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsMessage);
    return;
}

void
DiagServerEventHandler::replySupportedDTC(std::vector<DiagDtcData>& dtcInfos)
{
    DG_INFO << "DiagServerEventHandler::replySupportedDTC";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO);
    udsMessage.udsData.push_back(REPOT_TYPE_REPORTSUPPORTEDDTC);
    udsMessage.udsData.push_back(DTC_STATUS_AVAILABILITY_MASK);

    for (auto single : dtcInfos) {
        uint8_t dtcHighByte = single.dtc >> 16;
        uint8_t dtcMiddleByte = single.dtc>> 8;
        uint8_t dtcLowByte = single.dtc;
        uint8_t dtcStatus = single.dtcStatus;
        udsMessage.udsData.push_back(dtcHighByte);
        udsMessage.udsData.push_back(dtcMiddleByte);
        udsMessage.udsData.push_back(dtcLowByte);
        udsMessage.udsData.push_back(dtcStatus);
    }

    std::string temp;
    for (auto i : udsMessage.udsData) {
        char format[8]{0};
        snprintf(format, 8, "0x%02x", i);
        temp += " " + std::string(format);
    }
    DG_INFO << "DiagServerEventHandler::replySupportedDTC msg:" << temp;


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, udsMessage);
    return;
}

void
DiagServerEventHandler::replyControlDTCStatusType(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType)
{
    DG_INFO << "DiagServerEventHandler::replyControlDTCStatusType";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_CONTROL_DTC_SET);
    udsMessage.udsData.push_back(static_cast<std::underlying_type_t<DIAG_CONTROLDTCSTATUSTYPE>>(controlDtcStatusType));

    std::string temp;
    for (auto i : udsMessage.udsData) {
        temp += " " + std::to_string(i);
    }
    DG_INFO << "DiagServerEventHandler::replyControlDTCStatusType msg:" << temp;


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET, udsMessage);
    return;
}

void
DiagServerEventHandler::replyOutputDtcInfo(const bool result)
{
    DG_INFO << "DiagServerEventHandler::replyOutputDtcInfo result:" << result;
    DiagServerUdsMessage udsMessage;
    udsMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_ROUTINE_CONTROL);
    udsMessage.udsData.push_back(static_cast<int8_t>(result));


    DiagServerUdsMgr::getInstance()->sendPositiveResponse(DIAG_SERVER_SERVICE_REQUEST_OPC_ROUTINE_CONTROL, udsMessage);
    return;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
