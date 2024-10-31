/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid19.cpp is designed for diagnostic Read DTC Information.
 */

#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/event_manager/diag_server_event_handler.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid19.h"

namespace hozon {
namespace netaos {
namespace diag {


DiagServerUdsSid19::DiagServerUdsSid19()
{
}

DiagServerUdsSid19::~DiagServerUdsSid19()
{
}

void
DiagServerUdsSid19::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_INFO << "DiagServerUdsSid19::AnalyzeUdsMessage";
    m_udsMessage.Copy(udsMessage);

    if (udsMessage.udsData.empty() || 2 > udsMessage.udsData.size()) {
        DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage uds data invalid";
        // NRC 0x13
        sendNegative(kIncorrectMessageLengthOrInvalidFormat);
        return;
    }

    uint8_t sid = udsMessage.udsData[0];
    uint8_t subFuncId = udsMessage.udsData[1];
    DG_INFO << "DiagServerUdsSid19::AnalyzeUdsMessage sid:" << (int)sid << ",sfid:" << (int)subFuncId;

    switch (subFuncId) {
    case 0x01:
        {
            // 1.Check the data length
            if (3 != udsMessage.udsData.size()) {
                DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage 01sf uds data invalid,size:" << udsMessage.udsData.size();
                // NRC 0x13
                sendNegative(kIncorrectMessageLengthOrInvalidFormat);
                return;
            }

            // 2.get dtc status byte
            uint8_t dtcStatus = udsMessage.udsData[2];
            DiagServerEventHandler::getInstance()->reportNumberOfDTCByStatusMask(dtcStatus);
        }
        break;
    case 0x02:
        {
            // 1.Check the data length
            if (3 != udsMessage.udsData.size()) {
                DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage 02sf uds data invalid,size:" << udsMessage.udsData.size();
                // NRC 0x13
                sendNegative(kIncorrectMessageLengthOrInvalidFormat);
                return;
            }

            // 2.get dtc status byte
            uint8_t dtcStatus = udsMessage.udsData[2];
            DiagServerEventHandler::getInstance()->reportDTCByStatusMask(dtcStatus);
        }
        break;
    case 0x03:
        {
            // 1.Check the data length
            if (2 != udsMessage.udsData.size()) {
                DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage 02sf uds data invalid,size:" << udsMessage.udsData.size();
                sendNegative(kIncorrectMessageLengthOrInvalidFormat);
                return;
            }

            DiagServerEventHandler::getInstance()->reportDTCSnapshotIdentification();
        }
        break;
    case 0x04:
        {
            // 1.Check the data length
            if (6 != udsMessage.udsData.size()) {
                DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage 04sf uds data invalid,size:" << udsMessage.udsData.size();
                // NRC 0x13
                sendNegative(kIncorrectMessageLengthOrInvalidFormat);
                return;
            }

            // 2.get dtc and snapshot id
            uint8_t dtcHighByte = udsMessage.udsData[2];
            uint8_t dtcMidByte = udsMessage.udsData[3];
            uint8_t dtcLowByte = udsMessage.udsData[4];
            uint32_t dtcValue = dtcHighByte << 16 |  dtcMidByte << 8 | dtcLowByte;
            uint8_t dtcSsrId = udsMessage.udsData[5];
            DiagServerEventHandler::getInstance()->reportDTCSnapshotRecordByDTCNumber(dtcValue, dtcSsrId);
        }
        break;
    case 0x0A:
        {
            // 1.Check the data length
            if (2 != udsMessage.udsData.size()) {
                DG_ERROR << "DiagServerUdsSid19::AnalyzeUdsMessage 02sf uds data invalid,size:" << udsMessage.udsData.size();
                sendNegative(kIncorrectMessageLengthOrInvalidFormat);
                return;
            }

            DiagServerEventHandler::getInstance()->reportSupportedDTC();
        }
        break;
    default:
    // NRC 0x12
        sendNegative(kSubfunctionNotSupported);
        break;
    }
}

void
DiagServerUdsSid19::sendNegative(const DiagServerNrcErrc eNrc)
{
    DG_ERROR << "DiagServerUdsSid19::sendNegative nrc:" << eNrc;
    DiagServerUdsMessage resposeUdsMsg;
    resposeUdsMsg.Change(m_udsMessage);
    resposeUdsMsg.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    resposeUdsMsg.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO));
    resposeUdsMsg.udsData.push_back(eNrc);
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(resposeUdsMsg);
    return;
}

void
DiagServerUdsSid19::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage resposeUdsMsg;
    resposeUdsMsg.Change(m_udsMessage);
    resposeUdsMsg.udsData = udsMessage.udsData;

    DG_DEBUG << "resposeUdsMsg info: "
             << " id:" << resposeUdsMsg.id
             << " suppressPosRspMsgIndBit:" << resposeUdsMsg.suppressPosRspMsgIndBit
             << " pendingRsp:" << resposeUdsMsg.pendingRsp
             << " udsSa:" << resposeUdsMsg.udsSa
             << " udsTa:" << resposeUdsMsg.udsTa
             << " taType:" << resposeUdsMsg.taType;
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(resposeUdsMsg);
}

void
DiagServerUdsSid19::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage resposeUdsMsg;
    resposeUdsMsg.Change(m_udsMessage);
    resposeUdsMsg.udsData = udsMessage.udsData;

    DG_DEBUG << "resposeUdsMsg info: "
             << " id:" << resposeUdsMsg.id
             << " suppressPosRspMsgIndBit:" << resposeUdsMsg.suppressPosRspMsgIndBit
             << " pendingRsp:" << resposeUdsMsg.pendingRsp
             << " udsSa:" << resposeUdsMsg.udsSa
             << " udsTa:" << resposeUdsMsg.udsTa
             << " taType:" << resposeUdsMsg.taType;
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(resposeUdsMsg);
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
