/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid11.cpp is designed for diagnostic Ecu Reset.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid11.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid11::mtx_;

DiagServerUdsSid11::DiagServerUdsSid11()
{
}

DiagServerUdsSid11::~DiagServerUdsSid11()
{
}

void
DiagServerUdsSid11::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid11::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_ECU_RESET);

    // Check the data length
    if ((udsMessage.udsData.size() > 2)) {
        DG_ERROR << "DiagServerUdsSid11::AnalyzeUdsMessage error data size. udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_ECU_RESET, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid11::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    uint8_t sub_func = udsMessage.udsData.at(1);
    // Hard Reset
    if (sub_func == 0x01) {
        // TODO: Condition Check
        // Check Vehicle Speed Over 3km/h
        // Check Vehicle Status
        // Check Voltage Between 9V~16V
        // Notify Machine Start Hrad Reset(Reboot)
        responseMessage.udsData.push_back(sub_func);
    }

    PositiveResponse(responseMessage);
}

void
DiagServerUdsSid11::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_ECU_RESET, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid11::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid11::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_ECU_RESET));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
