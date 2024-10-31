/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid28.cpp is designed for diagnostic Communication Control.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/publish/diag_server_uds_pub.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid28.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid28::mtx_;

const uint DATA_LENGTH_WITHOUT_ADDRESS_SID_28 = 3;
const uint DATA_LENGTH_WITH_ADDRESS_SID_28 = 5;

DiagServerUdsSid28::DiagServerUdsSid28()
{
}

DiagServerUdsSid28::~DiagServerUdsSid28()
{
}

void
DiagServerUdsSid28::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid28::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_COMMUNICATION_CONTROL);

    // Check the data length
    if ((udsMessage.udsData.size() != 3) && (udsMessage.udsData.size() != 5)) {
        DG_ERROR << "DiagServerUdsSid28::AnalyzeUdsMessage error data size. needDataLength: " << DATA_LENGTH_WITHOUT_ADDRESS_SID_28 
                                                                                              << " or " << DATA_LENGTH_WITH_ADDRESS_SID_28
                                                                                              << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Check whether the sub-function supports
    uint8_t subFunction = udsMessage.udsData[1];
    if (subFunction > 0x05) {
        DG_ERROR << "DiagServerUdsSid28::AnalyzeUdsMessage subFunction not support. subFunction: " << UINT8_TO_STRING(subFunction);
        nrc_errc_ = DiagServerNrcErrc::kSubfunctionNotSupported;
        NegativeResponse(responseMessage);
        return;
    }

    // Check whether the communication type support
    uint8_t comm_type = udsMessage.udsData[2];
    if (comm_type < 0x1 || comm_type > 0x3) {
        DG_ERROR << "DiagServerUdsSid28::AnalyzeUdsMessage error communication type. comm_type: " << UINT8_TO_STRING(comm_type);
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    // Check address data length
    if (subFunction > 0x03) {
        // Check whether the address meets the requirements
        uint16_t address = udsMessage.udsData[3];
        address = (address << 8) | udsMessage.udsData[4];
        bool bCheckResult = true; // TO DO
        if (!bCheckResult) {
            DG_ERROR << "DiagServerUdsSid28::AnalyzeUdsMessage error address. address: " << UINT16_TO_STRING(address);
            nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
            NegativeResponse(responseMessage);
            return;
        }
    }

#ifdef BUILD_FOR_ORIN
    // notify mcu
    std::vector<uint8_t> data;
    data.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());
    data.insert(data.begin(), static_cast<uint8_t>(udsMessage.udsData.size()));
    DiagServerUdsPub::getInstance()->SendUdsEvent(data);
#endif
    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid28::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    responseMessage.udsData.push_back(subFunction);

    bool commControlResult = false;
    if (0x00 == subFunction) {
        commControlResult = EnableRxAndTx();
    }
    else if (0x01 == subFunction) {
        commControlResult = EnableRxAndDisableTx();
    }
    else if (0x02 == subFunction) {
        commControlResult = DisableRxAndEnableTx();
    }
    else if (0x03 == subFunction) {
        commControlResult = DisableRxAndTx();
    }
    else if (0x04 == subFunction) {
        commControlResult = EnableRxAndDisableTxWithAddress();
    }
    else if (0x05 == subFunction) {
        commControlResult = EnableRxAndTxWithAddress();
    }
    else {
        DG_WARN << "DiagServerUdsSid28::AnalyzeUdsMessage  invalid subFunction. subFunction: " << UINT8_TO_STRING(subFunction);
    }

    if (!commControlResult) {
        DG_ERROR << "DiagServerUdsSid28::AnalyzeUdsMessage dtc control failed. subFunction: " << UINT8_TO_STRING(subFunction);
        nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
        NegativeResponse(responseMessage);
        return;
    }

    PositiveResponse(responseMessage);
}

bool
DiagServerUdsSid28::EnableRxAndTx()
{
    DG_DEBUG << "DiagServerUdsSid28::EnableRxAndTx.";
    return true;
}

bool
DiagServerUdsSid28::EnableRxAndDisableTx()
{
    DG_DEBUG << "DiagServerUdsSid28::EnableRxAndDisableTx.";
    return true;
}

bool
DiagServerUdsSid28::DisableRxAndEnableTx()
{
    DG_DEBUG << "DiagServerUdsSid28::DisableRxAndEnableTx.";
    return true;
}

bool
DiagServerUdsSid28::DisableRxAndTx()
{
    DG_DEBUG << "DiagServerUdsSid28::DisableRxAndTx.";
    return true;
}

bool
DiagServerUdsSid28::EnableRxAndDisableTxWithAddress()
{
    DG_DEBUG << "DiagServerUdsSid28::EnableRxAndDisableTxWithAddress.";
    return true;
}

bool
DiagServerUdsSid28::EnableRxAndTxWithAddress()
{
    DG_DEBUG << "DiagServerUdsSid28::EnableRxAndTxWithAddress.";
    return true;
}

void
DiagServerUdsSid28::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid28::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid28::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
