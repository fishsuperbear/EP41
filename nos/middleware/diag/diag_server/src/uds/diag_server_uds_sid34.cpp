/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid34.cpp is designed for diagnostic Request for download.
 */

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid34.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid34::mtx_;

const uint MININUM_DATA_LENGTH_SID_34 = 4;

DiagServerUdsSid34::DiagServerUdsSid34()
{
}

DiagServerUdsSid34::~DiagServerUdsSid34()
{
}

void
DiagServerUdsSid34::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid34::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
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
    if (udsMessage.udsData.size() < MININUM_DATA_LENGTH_SID_34) {
        DG_ERROR << "DiagServerUdsSid34::AnalyzeUdsMessage error data size. mininumDataLength: " << MININUM_DATA_LENGTH_SID_34 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // Get data format
    // uint8_t dataFormat = udsMessage.udsData[1];

    // Check the file name length
    uint8_t memorySizeLength = static_cast<uint8_t>(udsMessage.udsData[2] >> 4);
    uint8_t memoryAddressLength = static_cast<uint8_t>(udsMessage.udsData[2] & 0xf);
    auto udsDataSize = udsMessage.udsData.size();
    if ((udsDataSize - 3) != (memorySizeLength + memoryAddressLength)) {
        DG_ERROR << "DiagServerUdsSid34::AnalyzeUdsMessage error data size. needDataLength: " << memorySizeLength + memoryAddressLength + 3 << " udsdata.size: " << udsMessage.udsData.size();
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    std::string fileName = "";
    fileName.assign(udsMessage.udsData.begin() + 3, udsMessage.udsData.begin() + 3 + memoryAddressLength);
    DG_INFO << "DiagServerUdsSid34::AnalyzeUdsMessage fileName: " << fileName;

    uint32_t dataSize = 0;
    int displacement = 0;
    for (uint i = (udsDataSize - 1); i >= (udsDataSize - memorySizeLength); i--) {
        dataSize += static_cast<uint32_t>(udsMessage.udsData[i] << displacement);
        displacement += 8;
    }
    
    DG_INFO << "DiagServerUdsSid34::AnalyzeUdsMessage dataSize: " << dataSize;

    // Check fileName and dataSize valid or not
    if (("" == fileName) || (0 == dataSize)) {
        DG_ERROR << "DiagServerUdsSid34::AnalyzeUdsMessage fileName or dataSize is invalid. fileName: " << fileName << " dataSize: " << dataSize;
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid34::AnalyzeUdsMessage request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    // file download
    bool bStartResult = DiagServerDataTransfer::getInstance()->StartFileDataDownload(fileName, dataSize);
    if (false == bStartResult) {
        DG_ERROR << "DiagServerUdsSid34::AnalyzeUdsMessage StartFileDataDownload failed.";
        nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
        NegativeResponse(responseMessage);
        return;
    }

    // file upload
    // bool bStartResult = DiagServerDataTransfer::getInstance()->StartFileDataUpload(fileName, dataSize);
    // if (false == bStartResult) {
    //     DG_ERROR << "DiagServerUdsSid34::AnalyzeUdsMessage StartFileDataUpload failed.";
    //     nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
    //     NegativeResponse(responseMessage);
    //     return;
    // }

    // file download
    std::vector<uint8_t> sizeVec;
    DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSCAPACITY, sizeVec);
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_DOWNLOAD);
    responseMessage.udsData.push_back(static_cast<uint8_t>(sizeVec.size()));
    for (auto& item : sizeVec) {
        responseMessage.udsData.push_back(item);
    }

    PositiveResponse(responseMessage);

    // file upload
    // std::vector<uint8_t> sizeVec;
    // DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSDATA, sizeVec);
    // responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_DOWNLOAD);
    // responseMessage.udsData.push_back(static_cast<uint8_t>(sizeVec.size()));
    // for (auto& item : sizeVec) {
    //     responseMessage.udsData.push_back(item);
    // }

    // PositiveResponse(responseMessage);
}

void
DiagServerUdsSid34::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid34::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid34::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
