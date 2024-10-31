/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid2e.cpp is designed for diagnostic Write DataByIdentifier.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid2E.h"
#include "diag/diag_server/include/info/diag_server_dynamic_info.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid2E::mtx_;

const uint MININUM_DATA_LENGTH_SID_2E = 4;

DiagServerUdsSid2E::DiagServerUdsSid2E()
: nrc_errc_(DiagServerNrcErrc::kConditionsNotCorrect)
{
}

DiagServerUdsSid2E::~DiagServerUdsSid2E()
{
}

void
DiagServerUdsSid2E::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid2E::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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

    // Check the data length
    if (udsMessage.udsData.size() < MININUM_DATA_LENGTH_SID_2E) {
        DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage error data size. mininumDataLength: " << MININUM_DATA_LENGTH_SID_2E << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Judge whether did support
    uint16_t did = udsMessage.udsData[1];
    did = (did << 8) | udsMessage.udsData[2];
    DiagAccessPermissionDataInfo accessPermission;
    bool bSupport = DiagServerConfig::getInstance()->QueryWriteAccessPermissionByDid(did, accessPermission);
    DiagServerSessionCode currentSession =  DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession();
    auto itrSession = find(accessPermission.allowedSessions.begin(), accessPermission.allowedSessions.end(), static_cast<uint8_t>(currentSession));
    if (!bSupport || (itrSession == accessPermission.allowedSessions.end())) {
        DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage not support currentSession: " << currentSession << " did: " << UINT16_TO_STRING(did);
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    // Check did data length
    uint16_t didDataLength = DiagServerConfig::getInstance()->QueryDidDataSize(did);
    if ((udsMessage.udsData.size() - 3) != didDataLength) {
        DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage error data size. needDataLength: " << didDataLength + 3 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Check security access
    uint8_t securityLevel = DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel();
    auto itrLevel = find(accessPermission.allowedSecurityLevels.begin(), accessPermission.allowedSecurityLevels.end(), static_cast<uint8_t>(DiagSecurityLevelId::DiagSecurityLevelId_Non));
    if (itrLevel == accessPermission.allowedSecurityLevels.end()) {
        itrLevel = find(accessPermission.allowedSecurityLevels.begin(), accessPermission.allowedSecurityLevels.end(), securityLevel);
    }

    if (itrLevel == accessPermission.allowedSecurityLevels.end()) {
        DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage error security level. securityLevel: " << securityLevel;
        nrc_errc_ = DiagServerNrcErrc::kSecurityAccessDenied;
        NegativeResponse(responseMessage);
        return;
    }

    std::vector<uint8_t> didData;
    didData.assign(udsMessage.udsData.begin() + 3, udsMessage.udsData.begin() + 3 + didDataLength);
    if (!WriteDidDataCheck(did, didData)) {
            DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage did: " << UINT16_TO_STRING(did) << " write data check failed.";
            nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
            NegativeResponse(responseMessage);
            return;
        }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceByDid(did, service, true);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid2E::AnalyzeUdsMessage request to external service did: " << UINT16_TO_STRING(did);
        RequestToExternalService(service, udsMessage);
        return;
    }

    bool bWriteResult = WriteDidData(did, didData);
    if (false == bWriteResult) {
        DG_ERROR << "DiagServerUdsSid2E::AnalyzeUdsMessage wirte didData error. did: " << UINT16_TO_STRING(did);
        nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
        NegativeResponse(responseMessage);
        return;
    }

    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_WRITE_DATA_IDENTIFIER);
    responseMessage.udsData.push_back(static_cast<uint8_t>((did >> 8) & 0xff));
    responseMessage.udsData.push_back(static_cast<uint8_t>(did & 0xff));
    PositiveResponse(responseMessage);
}

bool
DiagServerUdsSid2E::WriteDidData(const uint16_t did, const std::vector<uint8_t>& didData)
{
    DG_DEBUG << "DiagServerUdsSid2E::WriteDidData did: " << UINT16_TO_STRING(did);
    bool bResult = false;
    switch (did)
    {
        case 0xF170:
            bResult = DiagServerStoredInfo::getInstance()->WriteVehicleCfgWordData(didData);
            break;
        case 0xF190:
            bResult = DiagServerStoredInfo::getInstance()->WriteVinData(didData);
            break;
        case 0xF198:
            bResult = DiagServerStoredInfo::getInstance()->WriteTesterSNData(didData);
            break;
        case 0xF199:
            bResult = DiagServerStoredInfo::getInstance()->WriteProgrammingDateData(didData);
            break;
        case 0xF19D:
            bResult = DiagServerStoredInfo::getInstance()->WriteInstallDate(didData);
            break;
        case 0x900F:
            bResult = DiagServerStoredInfo::getInstance()->WriteEskNumber(didData);
            break;
        case 0x2910:
            bResult = DiagServerStoredInfo::getInstance()->WriteFactoryMode(didData);
            break;
        default:
            break;
    }

    return bResult;
}

bool
DiagServerUdsSid2E::WriteDidDataCheck(const uint16_t did, const std::vector<uint8_t>& didData)
{
    DG_DEBUG << "DiagServerUdsSid2E::WriteDidData did: " << UINT16_TO_STRING(did);
    DiagServerInfoDataType type = DiagServerInfoDataType::kHEX;
    switch (did) {
        case 0xF190:
            type = DiagServerInfoDataType::kNumberAndLetter;
            break;
        case 0xF198:
            type = DiagServerInfoDataType::kASCII;
            break;
        case 0xF199:
            type = DiagServerInfoDataType::kBCD;
            break;
        case 0xF19D:
            type = DiagServerInfoDataType::kBCD;
            break;
        case 0x900F:
            type = DiagServerInfoDataType::kASCII;
            break;
        default:
            break;
    }

    return DiagServerStoredInfo::getInstance()->DataCheck(type, didData);
}

void
DiagServerUdsSid2E::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsTa;
    responseMessage.udsTa = udsMessage.udsSa;
    responseMessage.taType = udsMessage.taType;

    uint8_t sub_func = 0xFF;
    std::vector<uint8_t> udsData;
    udsData.assign(udsMessage.udsData.begin() + 1, udsMessage.udsData.end());
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid2E::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid2E::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon