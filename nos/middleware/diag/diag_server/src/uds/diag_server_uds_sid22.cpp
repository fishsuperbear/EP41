/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid22.cpp is designed for diagnostic Read Data By Identifier.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid22.h"
#include "diag/diag_server/include/info/diag_server_dynamic_info.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid22::mtx_;

const uint MININUM_DATA_LENGTH_SID_22 = 3;

DiagServerUdsSid22::DiagServerUdsSid22()
: nrc_errc_(DiagServerNrcErrc::kConditionsNotCorrect)
{
}

DiagServerUdsSid22::~DiagServerUdsSid22()
{

}

void
DiagServerUdsSid22::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid22::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_READ_DATA_IDENTIFIER);

    // Check the data length
    if ((udsMessage.udsData.size() < MININUM_DATA_LENGTH_SID_22) || ((udsMessage.udsData.size() - 1) % 2 != 0)) {
        DG_ERROR << "DiagServerUdsSid22::AnalyzeUdsMessage error data size. mininumDataLength: " << MININUM_DATA_LENGTH_SID_22 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Check Whether the number of DIDs exceeds the maximum limit
    if ((udsMessage.udsData.size() - 1) / 2 > DIAG_SERVER_READ_DID_MAX_NUMBER) {
        DG_ERROR << "DiagServerUdsSid22::AnalyzeUdsMessage received DIDs exceeds the maximum limit.";
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    int didNumber = (udsMessage.udsData.size() -1) / 2;
    int offset = 1;
    uint16_t did = 0;
    if (1 == didNumber) {
        did = udsMessage.udsData[1];
        did = (did << 8) | udsMessage.udsData[2];
        if (!DidSupportAndSecurityCheck(did)) {
            NegativeResponse(responseMessage);
            return;
        }

        // Whether it is an request to external service
        std::vector<std::string> service;
        bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceByDid(did, service);
        if (bResult) {
            DG_DEBUG << "DiagServerUdsSid22::AnalyzeUdsMessage request to external service did: " << UINT16_TO_STRING(did);
            RequestToExternalService(service, udsMessage);
            return;
        }
    }

    for (int i = 0; i < didNumber; i++) {
        did = udsMessage.udsData[offset++];
        did = (did << 8) | udsMessage.udsData[offset++];
        if (!DidSupportAndSecurityCheck(did)) {
            NegativeResponse(responseMessage);
            return;
        }

        std::vector<uint8_t> vDidResultData;
        bool bResult = ReadDidData(did, vDidResultData);
        if (false == bResult) {
            DG_ERROR << "DiagServerUdsSid22::AnalyzeUdsMessage get didData error. did: " << UINT16_TO_STRING(did);
            nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
            NegativeResponse(responseMessage);
            return;
        }

        responseMessage.udsData.push_back(static_cast<uint8_t>((did >> 8) & 0xff));
        responseMessage.udsData.push_back(static_cast<uint8_t>(did & 0xff));
        responseMessage.udsData.insert(responseMessage.udsData.end(), vDidResultData.begin(), vDidResultData.end());
    }

    // Check the length of the acquired did data
    if (responseMessage.udsData.size() < 2) {
        DG_ERROR << "DiagServerUdsSid22::AnalyzeUdsMessage all dids not support.";
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    PositiveResponse(responseMessage);
}

bool
DiagServerUdsSid22::DidSupportAndSecurityCheck(const uint16_t did)
{
    DG_DEBUG << "DiagServerUdsSid22::DidSupportAndSecurityCheck did: " << UINT16_TO_STRING(did);
    // Judge whether did support
    DiagAccessPermissionDataInfo accessPermission;
    bool bSupport = DiagServerConfig::getInstance()->QueryReadAccessPermissionByDid(did, accessPermission);
    if (!bSupport) {
        DG_WARN << "DiagServerUdsSid22::DidSupportAndSecurityCheck not support did: " << UINT16_TO_STRING(did);
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        return false;
    }

    // Check security access
    DiagServerSessionCode currentSession =  DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetCurrentSession();
    uint8_t securityLevel = DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().GetSecurityLevel();
    auto itrSession = find(accessPermission.allowedSessions.begin(), accessPermission.allowedSessions.end(), static_cast<uint8_t>(currentSession));
    auto itrLevel = find(accessPermission.allowedSecurityLevels.begin(), accessPermission.allowedSecurityLevels.end(), static_cast<uint8_t>(DiagSecurityLevelId::DiagSecurityLevelId_Non));
    if (itrLevel == accessPermission.allowedSecurityLevels.end()) {
        itrLevel = find(accessPermission.allowedSecurityLevels.begin(), accessPermission.allowedSecurityLevels.end(), securityLevel);
    }

    if (itrSession == accessPermission.allowedSessions.end() || itrLevel == accessPermission.allowedSecurityLevels.end()) {
        DG_ERROR << "DiagServerUdsSid22::DidSupportAndSecurityCheck error session or security level. currentSession: " << currentSession << " securityLevel: " << securityLevel;
        nrc_errc_ = DiagServerNrcErrc::kSecurityAccessDenied;
        return false;
    }

    return true;
}

bool
DiagServerUdsSid22::ReadDidData(const uint16_t did, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid22::GetDidData did: " << UINT16_TO_STRING(did);
    bool bResult = false;
    std::vector<uint8_t> didData;
    // didData.push_back(static_cast<uint8_t>((did >> 8) & 0xff));
    // didData.push_back(static_cast<uint8_t>(did & 0xff));
    switch (did)
    {
        case 0xF170:
            bResult = DiagServerStoredInfo::getInstance()->ReadVehicleCfgWordData(didData);
            break;
        case 0xF188:
            bResult = DiagServerStoredInfo::getInstance()->ReadECUSWData(didData);
            break;
        case 0xF190:
            bResult = DiagServerStoredInfo::getInstance()->ReadVinData(didData);
            break;
        case 0xF198:
            bResult = DiagServerStoredInfo::getInstance()->ReadTesterSNData(didData);
            break;
        case 0xF199:
            bResult = DiagServerStoredInfo::getInstance()->ReadProgrammingDateData(didData);
            break;
        case 0x0107:
            bResult = DiagServerDynamicInfo::getInstance()->ReadInstallStatus(didData);
            break;
        case 0x0110:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuType(didData);
            break;
        case 0xF180:
            bResult = DiagServerStoredInfo::getInstance()->ReadBootSWId(didData);
            break;
        case 0xF186:
            bResult = DiagServerStoredInfo::getInstance()->ReadCurrDiagSession(didData);
            break;
        case 0xF187:
            bResult = DiagServerStoredInfo::getInstance()->ReadVehicleManufacturerSparePartNumber(didData);
            break;
        case 0xF1B0:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuSoftwareNumber(didData);
            break;
        case 0xF18A:
            bResult = DiagServerStoredInfo::getInstance()->ReadSystemSupplierId(didData);
            break;
        case 0xF18B:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuManufactureDate(didData);
            break;
        case 0xF18C:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuSerialNumber(didData);
            break;
        case 0xF191:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuHardwareVersion(didData);
            break;
        case 0xF19D:
            bResult = DiagServerStoredInfo::getInstance()->ReadInstallDate(didData);
            break;
        case 0xF1BF:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuHardwareNumber(didData);
            break;
        case 0xF1D0:
            bResult = DiagServerStoredInfo::getInstance()->ReadEcuSoftwareAssemblyPartNumber(didData);
            break;
        case 0x900F:
            bResult = DiagServerStoredInfo::getInstance()->ReadEskNumber(didData);
            break;
        case 0xF1E0:
            bResult = DiagServerStoredInfo::getInstance()->ReadAllSensorVersions(didData);
            break;
        case 0xF1E1:
            bResult = DiagServerStoredInfo::getInstance()->ReadOrinVersion(didData);
            break;
        case 0xF1E2:
            bResult = DiagServerStoredInfo::getInstance()->ReadSOCVersion(didData);
            break;
        case 0xF1E3:
            bResult = DiagServerStoredInfo::getInstance()->ReadMCUVersion(didData);
            break;
        case 0x2910:
            bResult = DiagServerStoredInfo::getInstance()->ReadFactoryMode(didData);
            break;
        case 0x0112:
            bResult = DiagServerDynamicInfo::getInstance()->ReadPowerSupplyVoltage(didData);
            break;
        case 0xE101:
            bResult = DiagServerDynamicInfo::getInstance()->ReadOdometerValue(didData);
            break;
        case 0xB100:
            bResult = DiagServerDynamicInfo::getInstance()->ReadVehicleSpeed(didData);
            break;
        case 0xD001:
            bResult = DiagServerDynamicInfo::getInstance()->ReadIgnitionStatus(didData);
            break;
        case 0xF020:
            bResult = DiagServerDynamicInfo::getInstance()->ReadTime(didData);
            break;
        case 0x8001:
            bResult = DiagServerDynamicInfo::getInstance()->ReadPKIApplyStatus(didData);
            break;
        case 0xF103:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASF30CameraCalibrationStatus(didData);
            break;
        case 0xF104:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASF120CameraCalibrationStatus(didData);
            break;
        case 0xF105:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFLCameraCalibrationStatus(didData);
            break;
        case 0xF106:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFRCameraCalibrationStatus(didData);
            break;
        case 0xF107:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASRLCameraCalibrationStatus(didData);
            break;
        case 0xF108:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASRRCameraCalibrationStatus(didData);
            break;
        case 0xF109:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASRearCameraCalibrationStatus(didData);
            break;
        case 0xF117:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASF30AndF120CameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF118:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASF120AndRLCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF119:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASF120AndRRCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF120:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFLAndRLCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF121:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFRAndRRCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF122:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFLAndRearCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF123:
            bResult = DiagServerDynamicInfo::getInstance()->ReadADASFRAndRearCameraCoordinatedCalibrationStatus(didData);
            break;
        case 0xF110:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASF30CameraCalibrationStatus(didData);
            break;
        case 0xF111:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASF120CameraCalibrationStatus(didData);
            break;
        case 0xF112:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASFLCameraCalibrationStatus(didData);
            break;
        case 0xF113:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASFRCameraCalibrationStatus(didData);
            break;
        case 0xF114:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRLCameraCalibrationStatus(didData);
            break;
        case 0xF115:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRRCameraCalibrationStatus(didData);
            break;
        case 0xF116:
            bResult = DiagServerDynamicInfo::getInstance()->ReadAfterSalesADASRearCameraCalibrationStatus(didData);
            break;
        case 0xF1C1:
        case 0xF1C2:
        case 0xF1C3:
        case 0xF1C4:
        case 0xF1C5:
        case 0xF1C6:
            bResult = DiagServerDynamicInfo::getInstance()->ReadMcuDidInfo(did, didData);
            break;
        default:
            break;
    }

    if (bResult) {
        for (uint i = 0; i < didData.size(); i++) {
            udsData.push_back(didData[i]);
        }
    }

    return bResult;
}

void
DiagServerUdsSid22::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid22::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid22::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
