/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid36.cpp is designed for diagnostic Transfer data.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid36.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid36::mtx_;

const uint DOWNLOAD_MININUM_DATA_LENGTH_SID_36 = 3;
const uint UPLOAD_DATA_LENGTH_SID_36 = 2;

DiagServerUdsSid36::DiagServerUdsSid36()
{
    DiagServerSessionMgr::getInstance()->RegisterSessionStatusListener([](DiagServerSessionCode session)->void {
        DG_DEBUG << "DiagServerUdsSid36::RegisterSessionStatusListener session status " << session;
        DiagServerDataTransfer::getInstance()->StopDataTransfer();
    });
}

DiagServerUdsSid36::~DiagServerUdsSid36()
{
}

void
DiagServerUdsSid36::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid36::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_TRANSFER_DATA, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid36::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    // Check the data length
    auto transferStatus = DiagServerDataTransfer::getInstance()->GetDataTransferStatus();
    if (DataTransferStatus::DOWNLOADING == transferStatus) {
        if (udsMessage.udsData.size() < DOWNLOAD_MININUM_DATA_LENGTH_SID_36) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error data size. download mininumDataLength: " << DOWNLOAD_MININUM_DATA_LENGTH_SID_36 << " udsdata.size: " << udsMessage.udsData.size();
            nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
            NegativeResponse(responseMessage);
            return;
        }
    }
    else if (DataTransferStatus::UPLOADING == transferStatus) {
        if (udsMessage.udsData.size() != UPLOAD_DATA_LENGTH_SID_36) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error data size. upload needDataLength: " << UPLOAD_DATA_LENGTH_SID_36 << " udsdata.size: " << udsMessage.udsData.size();
            nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
            NegativeResponse(responseMessage);
            return;
        }
    }

    // Check sequence
    if (DataTransferStatus::STANDBY == transferStatus) {
        DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error sequence. transferStatus: " << transferStatus;
        nrc_errc_ = DiagServerNrcErrc::kRequestSequenceError;
        NegativeResponse(responseMessage);
        return;
    }

    // Check data size
    uint64_t currentDataSize = DiagServerDataTransfer::getInstance()->GetDataSize();
    uint8_t currentBlockCounter = DiagServerDataTransfer::getInstance()->GetBlockSequenceCounter();
    if (currentBlockCounter < udsMessage.udsData[1]) {
        if ((udsMessage.udsData.size() - 2) > currentDataSize) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error request udsdata size. requestdata.size: " << udsMessage.udsData.size() - 2
                                                                                    << " currentDataSize: " << currentDataSize;
            nrc_errc_ = DiagServerNrcErrc::kTransferDataSuspended;
            NegativeResponse(responseMessage);
            return;
        }
    }

    // Check block sequence counter
    if (0 == udsMessage.udsData[1]) {
        if (0xFF != currentBlockCounter) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error request block counter. requestBlockCounter: " << UINT8_TO_STRING(udsMessage.udsData[1])
                                                                                       << " currentBlockCounter: " << UINT8_TO_STRING(currentBlockCounter);
            nrc_errc_ = DiagServerNrcErrc::kWrongBlockSequenceCounter;
            NegativeResponse(responseMessage);
            return;
        }
    }
    else {
        if ((udsMessage.udsData[1] != currentBlockCounter) && (udsMessage.udsData[1] != (currentBlockCounter + 1))) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage error request block counter. requestBlockCounter: " << UINT8_TO_STRING(udsMessage.udsData[1])
                                                                                    << " currentBlockCounter: " << UINT8_TO_STRING(currentBlockCounter);
            nrc_errc_ = DiagServerNrcErrc::kWrongBlockSequenceCounter;
            NegativeResponse(responseMessage);
            return;
        }
    }

    // Check request out of range
    if (udsMessage.udsData[1] != currentBlockCounter) {
        if (DataTransferStatus::COMPLETED == transferStatus) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage request out of range. transferStatus: " << transferStatus;
            nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
            NegativeResponse(responseMessage);
            return;
        }
    }

    if (udsMessage.udsData[1] == currentBlockCounter) {
        responseMessage.udsData.assign(uds_data_.begin(), uds_data_.end());
        PositiveResponse(responseMessage);
        return;
    }

    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_TRANSFER_DATA);
    responseMessage.udsData.push_back(static_cast<uint8_t>(udsMessage.udsData[1]));

    std::vector<uint8_t> dataBlock;
    if (DataTransferStatus::DOWNLOADING == transferStatus) {
        dataBlock.clear();
        dataBlock.assign(udsMessage.udsData.begin() + 2, udsMessage.udsData.end());
        bool bResult = DiagServerDataTransfer::getInstance()->WriteDataToFileByCounter(udsMessage.udsData[1], dataBlock);
        if (!bResult) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage WriteDataToFileByCounter failed.";
            return;
        }
    }
    else if (DataTransferStatus::UPLOADING == transferStatus) {
        dataBlock.clear();
        bool bResult = DiagServerDataTransfer::getInstance()->ReadDataBlockByCounter(udsMessage.udsData[1], dataBlock);
        if (!bResult) {
            DG_ERROR << "DiagServerUdsSid36::AnalyzeUdsMessage ReadDataBlockByCounter failed.";
            return;
        }

        for (auto& item : dataBlock) {
            responseMessage.udsData.push_back(static_cast<uint8_t>(item));
        }
    }
    else {
        return;
    }

    uds_data_.clear();
    uds_data_.assign(responseMessage.udsData.begin(), responseMessage.udsData.end());
    DG_INFO << "DiagServerUdsSid36::AnalyzeUdsMessage data transfer total packages count: " << DiagServerDataTransfer::getInstance()->GetTotalBlockCount() << " , current package: " << DiagServerDataTransfer::getInstance()->GetTransferBlockCount();
    PositiveResponse(responseMessage);
}

void
DiagServerUdsSid36::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_TRANSFER_DATA, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid36::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid36::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_TRANSFER_DATA));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
