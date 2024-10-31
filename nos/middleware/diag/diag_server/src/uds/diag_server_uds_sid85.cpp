/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid85.cpp is designed for diagnostic Control DTC Setting.
 */

#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/info/diag_server_stored_info.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid85.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid85::mtx_;

const uint DATA_LENGTH_SID_85 = 2;

DiagServerUdsSid85::DiagServerUdsSid85()
{
}

DiagServerUdsSid85::~DiagServerUdsSid85()
{
}

void
DiagServerUdsSid85::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid85::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_CONTROL_DTC_SET);

    // Check the data length
    if (udsMessage.udsData.size() != DATA_LENGTH_SID_85) {
        DG_ERROR << "DiagServerUdsSid85::AnalyzeUdsMessage error data size. needDataLength: " << DATA_LENGTH_SID_85 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Check whether the sub-function supports
    uint8_t subFunction = udsMessage.udsData[1];
    if ((0x01 != subFunction) && (0x02 != subFunction)) {
        DG_ERROR << "DiagServerUdsSid85::AnalyzeUdsMessage subFunction not support. subFunction: " << UINT8_TO_STRING(subFunction);
        nrc_errc_ = DiagServerNrcErrc::kSubfunctionNotSupported;
        NegativeResponse(responseMessage);
        return;
    }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid85::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    if (0x01 == subFunction) {
        DG_DEBUG << "DiagServerUdsSid85::AnalyzeUdsMessage sw on.";
        DiagServerEventHandler::getInstance()->controlDTCStatusType(DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOn);
    }
    else if (0x02 == subFunction) {
        DG_DEBUG << "DiagServerUdsSid85::AnalyzeUdsMessage sw off.";
        DiagServerEventHandler::getInstance()->controlDTCStatusType(DIAG_CONTROLDTCSTATUSTYPE::kDTCSettingOff);
    }
    else {
        DG_WARN << "DiagServerUdsSid85::AnalyzeUdsMessage invalid subFunction. subFunction: " << UINT8_TO_STRING(subFunction);
    }

    // save dtc control status to cfg
    if (0x01 == subFunction || 0x02 == subFunction) {
        DiagServerStoredInfo::getInstance()->saveControlDtcStatusToCFG(subFunction);
    }

    responseMessage.udsData.push_back(subFunction);
    PositiveResponse(responseMessage);
}

void
DiagServerUdsSid85::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid85::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid85::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
