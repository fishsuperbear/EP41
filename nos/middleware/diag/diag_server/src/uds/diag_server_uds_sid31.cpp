/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid31.cpp is designed for diagnostic Routine Control.
 */

#include<algorithm>

#include "diag/diag_server/include/uds/diag_server_uds_sid31.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"
#include "diag_server_transport_cm.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid31::mtx_;

const uint MININUM_DATA_LENGTH_SID_31 = 4;

DiagServerUdsSid31::DiagServerUdsSid31()
: nrc_errc_(DiagServerNrcErrc::kConditionsNotCorrect)
{
    DiagServerSessionMgr::getInstance()->RegisterSessionStatusListener([](DiagServerSessionCode session)->void {
        // TODO: listen session status
        DG_DEBUG << "DiagServerUdsSid31::RegisterSessionStatusListener session status " << session;
    });

    InitRidStatusMap();
}

DiagServerUdsSid31::~DiagServerUdsSid31()
{

}

void
DiagServerUdsSid31::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid31::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
                                                                       <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                       <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);

    std::lock_guard<std::mutex> lck(mtx_);
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsTa;
    responseMessage.udsTa = udsMessage.udsSa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_ROUTINE_CONTROL);

    // Check the data length < 4
    if (udsMessage.udsData.size() < MININUM_DATA_LENGTH_SID_31) {
        // NRC 0x13
        DG_ERROR << "DiagServerUdsSid31 | length error,length < 4!";
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    uint8_t sub_func = udsMessage.udsData[1];
    // uint32_t ret = 0x00;
    // bool support = false;
    uint16_t rid = udsMessage.udsData[2];
    rid = (rid << 8) | udsMessage.udsData[3];

    // check rid supported in active session

    // check security

    // check rid
    if (!CheckRid(rid)) {
        DG_ERROR << "DiagServerUdsSid31 | rid not supported!";
        // NRC 0x31
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    // check subfunc
    if (!CheckSubFunction(rid, sub_func)) {
        DG_ERROR << "DiagServerUdsSid31 | sub function not supported!";
        // NRC 0x12
        nrc_errc_ = DiagServerNrcErrc::kSubfunctionNotSupported;
        NegativeResponse(responseMessage);
        return;
    }

    responseMessage.udsData.push_back(sub_func);
    responseMessage.udsData.push_back(static_cast<uint8_t>((rid >> 8) & 0xff));
    responseMessage.udsData.push_back(static_cast<uint8_t>(rid & 0xff));

    // check active session
    if (!CheckActiveSession(rid)) {
        DG_ERROR << "DiagServerUdsSid31 | active session not supported!";
        // NRC 0x7F
        nrc_errc_ = DiagServerNrcErrc::kServiceNotSupportedInActiveSession;
        NegativeResponse(responseMessage);
        return;
    }

    // check security
    if (!CheckSecurity(rid)) {
        DG_ERROR << "DiagServerUdsSid31 | security not supported!";
        // NRC 0x33
        nrc_errc_ = DiagServerNrcErrc::kSecurityAccessDenied;
        NegativeResponse(responseMessage);
        return;
    }

    // check total length
    // if (!CheckTotalLength(rid, sub_func, (udsMessage.udsData.size() - 4))) {
    //     DG_ERROR << "DiagServerUdsSid31 | total length error!";
    //     // NRC 0x13
    //     nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
    //     NegativeResponse(responseMessage);
    //     return;
    // }

    std::vector<uint8_t> udsData;
    if (udsMessage.udsData.size() > MININUM_DATA_LENGTH_SID_31) {
        udsData.assign(udsMessage.udsData.begin() + MININUM_DATA_LENGTH_SID_31, udsMessage.udsData.end());
    }

    // save fc88 calibrate status to cfg
    if (Sid31SubFuncId_Start == sub_func && 0xFC88 == rid) {
        DiagServerStoredInfo::getInstance()->saveCalibrateStatusToCFG(udsData);
    }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceByRid(rid, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid31 | external service rid: " << UINT16_TO_STRING(rid);
        RequestToExternalService(service, udsMessage);
        return;
    }

    bool bSequenceResult = false;
    if (Sid31SubFuncId_Start == sub_func) {
        bSequenceResult = Start(rid, udsData);
    }
    else if (Sid31SubFuncId_Stop == sub_func) {
        bSequenceResult = Stop(rid, udsData);
    }
    else if (Sid31SubFuncId_Result == sub_func) {
        bSequenceResult = Result(rid, udsData);
    }

    // check sequence
    if (!bSequenceResult) {
        DG_ERROR << "DiagServerUdsSid31 | sequence error!";
        // NRC 0x24
        nrc_errc_ = DiagServerNrcErrc::kRequestSequenceError;
        NegativeResponse(responseMessage);
        return;
    }

    for (auto item : udsData) {
        responseMessage.udsData.push_back(item);
    }

    PositiveResponse(responseMessage);

}

void
DiagServerUdsSid31::InitRidStatusMap()
{
    DG_INFO << "DiagServerUdsSid31::InitRidStatusMap.";
    std::vector<uint16_t> ridList;
    DiagServerConfig::getInstance()->QueryRidSupportList(ridList);
    for (auto& item : ridList) {
        rid_status_map_.insert(std::make_pair(item, Sid31RidStatus_Default));
    }
}

bool
DiagServerUdsSid31::CheckRid(const uint16_t rid)
{
    return DiagServerConfig::getInstance()->QueryRidSupport(rid);
}

bool
DiagServerUdsSid31::CheckSubFunction(const uint16_t rid, const uint8_t sub_func_id)
{
    bool ret = false;
    DiagRidDataInfo data;
    DiagServerConfig::getInstance()->QueryRidDataInfo(rid, data);
    for (auto & item : data.ridSubFunctions) {
        if (item.id == sub_func_id) {
            ret = true;
        }
    }

    return ret;
}

bool
DiagServerUdsSid31::CheckActiveSession(const uint16_t rid)
{
    DiagAccessPermissionDataInfo data;
    DiagServerConfig::getInstance()->QueryAccessPermissionByRid(rid, data);
    DiagServerSessionCode current_session = DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession();
    for (auto & item : data.allowedSessions) {
        if (item == current_session) {
            return true;
        }
    }

    return false;
}

bool
DiagServerUdsSid31::CheckSecurity(const uint16_t rid)
{
    DiagAccessPermissionDataInfo data;
    DiagServerConfig::getInstance()->QueryAccessPermissionByRid(rid, data);
    uint8_t security_level = DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel();
    for (auto & item : data.allowedSecurityLevels) {
        if (item == 0) {
            return true;
        }

        if (item == security_level) {
            return true;
        }
    }

    return false;
}


bool
DiagServerUdsSid31::CheckTotalLength(const uint16_t rid, const uint8_t sub_func_id, const size_t length, const bool isReply)
{
    bool ret = false;
    ret = DiagServerConfig::getInstance()->QueryRidDataLenBySubFunction(rid, sub_func_id, length, isReply);
    return ret;
}

bool
DiagServerUdsSid31::CheckSequence(const uint16_t rid, bool bStart)
{
    // sequence check
    auto itr = rid_status_map_.find(rid);
    if (itr == rid_status_map_.end()) {
        DG_ERROR << "DiagServerUdsSid31::CheckSequence not support rid: " << UINT16_TO_STRING(rid);
        return false;
    }

    if (bStart) {
        if (DiagServerConfig::getInstance()->QueryRidMultiStartSupport(rid)) {
            return true;
        }

        if (Sid31RidStatus_Started == itr->second) {
            DG_ERROR << "DiagServerUdsSid31::CheckSequence sequence error.";
            return false;
        }
    }
    else {
        if (Sid31RidStatus_Started != itr->second) {
            DG_ERROR << "DiagServerUdsSid31::CheckSequence sequence error.";
            return false;
        }
    }

    return true;
}

void
DiagServerUdsSid31::UpdateRidStatus(const uint16_t rid, const Sid31RidStatus& status)
{
    // update status
    auto itr = rid_status_map_.find(rid);
    if (itr == rid_status_map_.end()) {
        DG_ERROR << "DiagServerUdsSid31::UpdateRidStatus not support rid: " << UINT16_TO_STRING(rid);
        return;
    }

    itr->second = status;
}

bool
DiagServerUdsSid31::Start(const uint16_t rid, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::Start | rid: " << UINT16_TO_STRING(rid) << ", reqCmd size: " << udsData.size();
    // sequence check
    if (!CheckSequence(rid, true)) {
        DG_ERROR << "DiagServerUdsSid31::Start sequence error.";
        return false;
    }

    switch (rid)
    {
        case 0x6140:
            StartTest(udsData);
            break;
        case 0x0203:
            StartCheckProgrammingPreconditions(udsData);
            break;
        case 0x0205:
            StartInstallation(udsData);
            break;
        case 0x0206:
            StartSwitchRunningPartition(udsData);
            break;
        case 0xFF01:
            StartCheckProgrammingDependencies(udsData);
            break;
        case 0xD000:
            StartReportFaultOccur(udsData);
            break;
        case 0xD001:
            StartReportFaultRecover(udsData);
            break;
        case 0xD005:
            StartRefreshFaultFile(udsData);
            break;
        default:
            break;
    }

    UpdateRidStatus(rid, Sid31RidStatus_Started);
    return true;
}

void
DiagServerUdsSid31::StartTest(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartTest.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    udsData.emplace_back(0x01);
}

void
DiagServerUdsSid31::StartCheckProgrammingPreconditions(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartCheckProgrammingPreconditions.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StartInstallation(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartInstallation.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StartSwitchRunningPartition(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartSwitchRunningPartition.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StartCheckProgrammingDependencies(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartCheckProgrammingDependencies.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
    udsData.emplace_back(reqCmd[8]);
    udsData.emplace_back(reqCmd[9]);
}

void
DiagServerUdsSid31::StartReportFaultOccur(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartReportFaultOccur.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StartReportFaultRecover(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartReportFaultRecover.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StartRefreshFaultFile(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StartRefreshFaultFile.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

bool
DiagServerUdsSid31::Stop(const uint16_t rid, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::Stop | rid: " << UINT16_TO_STRING(rid) << ", reqCmd size: " << udsData.size();
    // sequence check
    if (!CheckSequence(rid)) {
        DG_ERROR << "DiagServerUdsSid31::Stop sequence error.";
        return false;
    }

    switch (rid)
    {
        case 0x6140:
            StopTest(udsData);
            break;
        case 0xD000:
            StopReportFaultOccur(udsData);
            break;
        case 0xD001:
            StopReportFaultRecover(udsData);
            break;
        default:
            break;
    }

    UpdateRidStatus(rid, Sid31RidStatus_Stopped);
    return true;
}

void
DiagServerUdsSid31::StopTest(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StopTest.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    udsData.emplace_back(0x04);
}

void
DiagServerUdsSid31::StopReportFaultOccur(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StopReportFaultOccur.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::StopReportFaultRecover(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::StopReportFaultRecover.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

bool
DiagServerUdsSid31::Result(const uint16_t rid, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::Result | rid: " << UINT16_TO_STRING(rid) << ", reqCmd size: " << udsData.size();
    switch (rid)
    {
        case 0x6140:
            ResultTest(udsData);
            break;
        case 0xD002:
            ResultQueryCurrentFault(udsData);
            break;
        case 0xD003:
            ResultQueryDtcByFault(udsData);
            break;
        case 0xD004:
            ResultQueryFaultByDtc(udsData);
            break;
        default:
            break;
    }

    return true;
}

void
DiagServerUdsSid31::ResultTest(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::ResultTest.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    udsData.emplace_back(0x03);
}

void
DiagServerUdsSid31::ResultQueryCurrentFault(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::ResultQueryCurrentFault.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::ResultQueryDtcByFault(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::ResultQueryDtcByFault.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);

}

void
DiagServerUdsSid31::ResultQueryFaultByDtc(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid31::ResultQueryFaultByDtc.";
    std::vector<uint8_t> reqCmd;
    reqCmd.assign(udsData.begin(), udsData.end());
    udsData.clear();
    // TODO
    udsData.emplace_back(0x02);
}

void
DiagServerUdsSid31::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsTa;
    responseMessage.udsTa = udsMessage.udsSa;
    responseMessage.taType = udsMessage.taType;

    uint8_t sub_func = udsMessage.udsData[1];
    std::vector<uint8_t> udsData;
    udsData.assign(udsMessage.udsData.begin() + 2, udsMessage.udsData.end());
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_ROUTINE_CONTROL, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid31::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid31::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_ROUTINE_CONTROL));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
