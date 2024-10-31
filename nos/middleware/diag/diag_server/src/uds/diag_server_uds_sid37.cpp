/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid37.cpp is designed for diagnostic Request Transfer Exit.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid37.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid37::mtx_;

const uint DATA_LENGTH_SID_37 = 1;

DiagServerUdsSid37::DiagServerUdsSid37()
{
}

DiagServerUdsSid37::~DiagServerUdsSid37()
{
}

void
DiagServerUdsSid37::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid37::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid37::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    // Check sequence
    auto transferStatus = DiagServerDataTransfer::getInstance()->GetDataTransferStatus();
    if (DataTransferStatus::STANDBY == transferStatus) {
        DG_ERROR << "DiagServerUdsSid37::AnalyzeUdsMessage error sequence. transferStatus: " << transferStatus;
        nrc_errc_ = DiagServerNrcErrc::kRequestSequenceError;
        NegativeResponse(responseMessage);
        return;
    }

    // Check transmission is completed
    if (DataTransferStatus::COMPLETED != transferStatus) {
        DG_ERROR << "DiagServerUdsSid37::AnalyzeUdsMessage transmission not completed. transferStatus: " << transferStatus;
        nrc_errc_ = DiagServerNrcErrc::kGeneralProgrammingFailure;
        NegativeResponse(responseMessage);
        return;
    }

    // Check the data length
    if (udsMessage.udsData.size() != DATA_LENGTH_SID_37) {
        DG_ERROR << "DiagServerUdsSid37::AnalyzeUdsMessage error data size. needDataLength: " << DATA_LENGTH_SID_37 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    bool bStopResult = DiagServerDataTransfer::getInstance()->StopDataTransfer();
    if (false == bStopResult) {
        DG_ERROR << "DiagServerUdsSid37::AnalyzeUdsMessage StopDataTransfer failed.";
        nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
        NegativeResponse(responseMessage);
        return;
    }

    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_TRANSFER_EXIT);
    PositiveResponse(responseMessage);
}

void
DiagServerUdsSid37::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid37::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid37::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
