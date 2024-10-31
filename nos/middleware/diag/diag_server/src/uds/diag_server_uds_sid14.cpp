/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid14.cpp is designed for diagnostic Clear dtc.
 */

#include "diag/common/include/format.h"
#include "diag/diag_server/include/common/diag_server_def.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid14.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid14::mtx_;

const uint DATA_LENGTH_SID_14 = 4;

DiagServerUdsSid14::DiagServerUdsSid14()
{
}

DiagServerUdsSid14::~DiagServerUdsSid14()
{
}

void
DiagServerUdsSid14::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid14::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
                                                                       <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                       <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    std::lock_guard<std::mutex> lck(mtx_);
    m_udsMessage.Copy(udsMessage);
    DiagServerUdsMessage responseMessage;
    responseMessage.Copy(udsMessage);

    // Check the data length
    if (udsMessage.udsData.size() != DATA_LENGTH_SID_14) {
        DG_ERROR << "DiagServerUdsSid14::AnalyzeUdsMessage error data size. needDataLength: " << DATA_LENGTH_SID_14 << " udsdata.size: " << udsMessage.udsData.size();
        DiagServerUdsMessage udsNegativeMsg;
        udsNegativeMsg.udsData.push_back(DiagServerNrcErrc::kNegativeHead);
        udsNegativeMsg.udsData.push_back(DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR);
        udsNegativeMsg.udsData.push_back(DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat); // NRC 0x13
        NegativeResponse(udsNegativeMsg);
        return;
    }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid14::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    // uint8_t sid = udsMessage.udsData[0];
    uint8_t groupHighByte = udsMessage.udsData[1];
    uint8_t groupMidByte = udsMessage.udsData[2];
    uint8_t groupLowByte = udsMessage.udsData[3];
    uint32_t dtcGroup = 0x00FFFFFF & (groupHighByte << 16 | groupMidByte << 8 | groupLowByte);
    DG_DEBUG << Format("DiagServerUdsSid14::AnalyzeUdsMessage dtcGroup:{}", dtcGroup);

    DiagServerEventHandler::getInstance()->clearDTCInformation(dtcGroup);
    return;
}

void
DiagServerUdsSid14::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.Change(udsMessage);
    uint8_t sub_func = 0xFF;
    std::vector<uint8_t> udsData;
    udsData.assign(udsMessage.udsData.begin() + 1, udsMessage.udsData.end());
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid14::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.Change(m_udsMessage);
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REPLY_OPC_DTC_CLEAR));


    DG_DEBUG << "resposeUdsMsg info: "
             << " id:" << responseMessage.id
             << " suppressPosRspMsgIndBit:" << responseMessage.suppressPosRspMsgIndBit
             << " pendingRsp:" << responseMessage.pendingRsp
             << " udsSa:" << responseMessage.udsSa
             << " udsTa:" << responseMessage.udsTa
             << " taType:" << responseMessage.taType;
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid14::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.Change(m_udsMessage);
    responseMessage.udsData.insert(responseMessage.udsData.end(), udsMessage.udsData.begin(), udsMessage.udsData.end());

    DG_DEBUG << "resposeUdsMsg info: "
             << " id:" << responseMessage.id
             << " suppressPosRspMsgIndBit:" << responseMessage.suppressPosRspMsgIndBit
             << " pendingRsp:" << responseMessage.pendingRsp
             << " udsSa:" << responseMessage.udsSa
             << " udsTa:" << responseMessage.udsTa
             << " taType:" << responseMessage.taType;
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
