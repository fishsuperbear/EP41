/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid38.cpp is designed for diagnostic Request FileT ransfer Replace File.
 */

#include "diag/diag_server/include/uds/diag_server_uds_sid38.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
// #include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/datatransfer/diag_server_data_transfer.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include <algorithm>

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid38::mtx_;

DiagServerUdsSid38::DiagServerUdsSid38()
{
}

DiagServerUdsSid38::~DiagServerUdsSid38()
{
}

void
DiagServerUdsSid38::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid38::AnalyzeUdsMessage sa: " << udsMessage.udsSa << " ta: " << udsMessage.udsTa << " udsData: " << UINT8_VEC_TO_STRING_P(udsMessage.udsData, udsMessage.udsData.size());
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsTa;
    responseMessage.udsTa = udsMessage.udsSa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_FILE_TRANSFER);

    // check the data length < 5
    if (udsMessage.udsData.size() < 0x05) {
        DG_ERROR << "DiagServerUdsSid38 | length error,length < 5!";
        // NRC 0x13
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    std::vector<uint8_t> udsData;
    udsData.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());
    // check total length
    if (!CheckTotalLength(udsData)) {
        DG_ERROR << "DiagServerUdsSid38 | total length error!";
        // NRC 0x13
        nrc_errc_ = DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat;
        NegativeResponse(responseMessage);
        return;
    }

    // check outOfRange
    if (!CheckOutOfRange(udsData)) {
        DG_ERROR << "DiagServerUdsSid38 | outOfRange error!";
        // NRC 0x31
        nrc_errc_ = DiagServerNrcErrc::kRequestOutOfRange;
        NegativeResponse(responseMessage);
        return;
    }

    // check Memory TODO
    // NRC 0x70

    // Whether it is an request to external service
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryExternalServiceBySid(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT, service);
    if (bResult) {
        DG_DEBUG << "DiagServerUdsSid38| request to external service.";
        RequestToExternalService(service, udsMessage);
        return;
    }

    if (!DealwithRequest(udsData)) {
        DG_ERROR << "DiagServerUdsSid38 | condition not correct!";
        // NRC 0x22
        nrc_errc_ = DiagServerNrcErrc::kConditionsNotCorrect;
        NegativeResponse(responseMessage);
        return;
    }

    for (auto item : udsData) {
        responseMessage.udsData.push_back(item);
    }

    PositiveResponse(responseMessage);
}

bool
DiagServerUdsSid38::DealwithRequest(std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid38::DealwithRequest udsData.size: " << udsData.size() << " udsData: " << UINT8_VEC_TO_STRING(udsData);
    bool bResult = true;
    std::vector<uint8_t> transfer;
    DiagSid38ParameterInfo parInfo;
    GetParameter(udsData, parInfo);

    udsData.clear();
    udsData.emplace_back(parInfo.opt_mode);
    if (parInfo.opt_mode == ModeOfOperation_AddFile || parInfo.opt_mode == ModeOfOperation_ReplaceFile ) {
        // compare Memory TODO
        bResult = DiagServerDataTransfer::getInstance()->StartFileDataDownload(parInfo.filePathAndName, parInfo.fileSize);
        if (!bResult) {
            DG_ERROR << "DiagServerUdsSid38::DealwithRequest StartFileDataDownload failed";
            return bResult;
        }

        DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSCAPACITY, transfer);
        udsData.emplace_back(transfer.size());
        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }

        udsData.emplace_back(parInfo.dataFormatIdentifier);
    }
    else if (parInfo.opt_mode == ModeOfOperation_DelFile) {
        remove(parInfo.filePathAndName.c_str());
    }
    else if (parInfo.opt_mode == ModeOfOperation_ReadFile) {
        bResult = DiagServerDataTransfer::getInstance()->StartFileDataUpload(parInfo.filePathAndName);
        if (!bResult) {
            DG_ERROR << "DiagServerUdsSid38::DealwithRequest StartFileDataUpload failed";
            return bResult;
        }

        DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSCAPACITY, transfer);
        udsData.emplace_back(transfer.size());
        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }

        udsData.emplace_back(parInfo.dataFormatIdentifier);
        transfer.clear();
        DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSDATA, transfer);
        uint16_t size = transfer.size();
        udsData.emplace_back(static_cast<uint8_t>(size >> 8));
        udsData.emplace_back(static_cast<uint8_t>(size));
        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }

        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }
    }
    else if (parInfo.opt_mode == ModeOfOperation_ReadDir) {
        bResult = DiagServerDataTransfer::getInstance()->StartFileDataUpload(parInfo.filePath);
        if (!bResult) {
            DG_ERROR << "DiagServerUdsSid38::DealwithRequest StartFileDataUpload failed";
            return bResult;
        }

        DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSCAPACITY, transfer);
        udsData.emplace_back(transfer.size());
        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }

        udsData.emplace_back(parInfo.dataFormatIdentifier);
        transfer.clear();
        DiagServerDataTransfer::getInstance()->GetSizeToVecWithType(DataTransferSizeType::TRANSDATA, transfer);
        uint16_t size = transfer.size();
        udsData.emplace_back(static_cast<uint8_t>(size >> 8));
        udsData.emplace_back(static_cast<uint8_t>(size));
        for (auto & item : transfer) {
            udsData.emplace_back(item);
        }
    }

    return bResult;
}

bool
DiagServerUdsSid38::CheckTotalLength(const std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid38::CheckTotalLength!";
    bool ret = false;
    DiagSid38ParameterInfo parInfo;
    GetParameter(udsData, parInfo);
    ret = (udsData.size() == parInfo.totalSize) ? true : false;
    if (!ret) {
        DG_ERROR << "DiagServerUdsSid38::CheckTotalLength | length error: "<< udsData.size() <<", expected length: " << parInfo.totalSize;
    }

    return ret;
}

bool
DiagServerUdsSid38::CheckOutOfRange(const std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerUdsSid38::CheckOutOfRange!";
    bool ret = false;
    DiagSid38ParameterInfo parInfo;
    GetParameter(udsData, parInfo);

    ret = (parInfo.filePathAndNameLength > 0x00) ? true : false;
    if (parInfo.opt_mode == ModeOfOperation_AddFile || parInfo.opt_mode == ModeOfOperation_ReplaceFile ) {
        size_t temp = 0;
        for (int i = 0; i < parInfo.fileSizeParameterLength; ++i) {
            temp = (i == 0) ? parInfo.fileSizeUnCompressed[i] : ((temp << 8) | parInfo.fileSizeUnCompressed[i]);
        }

        ret &= (temp > 0x00) ? true : false;
        ret &= (parInfo.dataFormatIdentifier == 0x00) ? true : false;
        ret &= (parInfo.fileSizeParameterLength > 0x00) ? true : false;
        ret &= (parInfo.fileSizeUnCompressed == parInfo.fileSizeCompressed) ? true : false;
        ret &= (CheckPath(0x02, parInfo.filePath)) ? true : false;
    }
    else if (parInfo.opt_mode == ModeOfOperation_DelFile) {
        ret &= (CheckPath(0x02, parInfo.filePath)) ? true : false;
    }
    else if (parInfo.opt_mode == ModeOfOperation_ReadFile) {
        ret &= (parInfo.dataFormatIdentifier == 0x00) ? true : false;
        ret &= (CheckPath(0x01, parInfo.filePath)) ? true : false;
    }
    else if (parInfo.opt_mode == ModeOfOperation_ReadDir) {
        ret |= (CheckPath(0x01, parInfo.filePath)) ? true : false;
        ret |= (CheckPath(0x02, parInfo.filePath)) ? true : false;
    }

    if (!ret) {
        DG_DEBUG << "DiagServerUdsSid38::CheckOutOfRange opt_mode: " << UINT8_TO_STRING(parInfo.opt_mode)
                 << " filePathAndNameLength: " << UINT16_TO_STRING(parInfo.filePathAndNameLength)
                 << " filePathAndName: " << parInfo.filePathAndName
                 << " dataFormatIdentifier: " << UINT8_TO_STRING(parInfo.dataFormatIdentifier)
                 << " fileSizeParameterLength: " << UINT8_TO_STRING(parInfo.fileSizeParameterLength)
                 << " fileSizeUnCompressed: " << UINT8_VEC_TO_STRING_P(parInfo.fileSizeUnCompressed, parInfo.fileSizeUnCompressed.size())
                 << " fileSizeCompressed: " << UINT8_VEC_TO_STRING_P(parInfo.fileSizeCompressed, parInfo.fileSizeCompressed.size())
                 << " fileSize: " << parInfo.fileSize
                 << " totalSize: " << parInfo.totalSize;
    }

    return ret;
}

bool
DiagServerUdsSid38::CheckPath(const uint8_t configId, const std::string& filePath)
{
    bool ret = false;
    DiagTransferConfigDataInfo dataInfo;
    ret = DiagServerConfig::getInstance()->QueryDataTransferConfig(configId, dataInfo);
    if (!ret) {
        DG_ERROR << "DiagServerUdsSid38::CheckPath error. configId: " << configId;
        return ret;
    }

    for (auto & item : dataInfo.pathWhiteList) {
        if (item == filePath) {
            ret = true;
            break;
        }
        else {
            ret = false;
        }
    }

    if (!ret) {
        DG_ERROR << "DiagServerUdsSid38::CheckPath filePath: " << filePath << " Not supported on the whitelist!";
    }

    return ret;
}

void
DiagServerUdsSid38::GetParameter(std::vector<uint8_t> data, DiagSid38ParameterInfo& infoData)
{
    DG_DEBUG << "DiagServerUdsSid38::GetParameter";
    if (data.empty()) {
        DG_ERROR << "DiagServerUdsSid38::GetParameter data is empty!";
        return;
    }

    infoData.opt_mode = data[1];
    infoData.filePathAndNameLength = data[2];
    infoData.filePathAndNameLength = (infoData.filePathAndNameLength << 8) | data[3];
    std::vector<uint8_t> vec;
    for (int i = 0; i < infoData.filePathAndNameLength; ++i) {
        vec.emplace_back(data[i + 4]);
    }

    if (infoData.opt_mode == ModeOfOperation_AddFile || infoData.opt_mode == ModeOfOperation_ReplaceFile ) {
        GetPath(vec, infoData.filePathAndName, infoData.filePath);
        infoData.dataFormatIdentifier = data[infoData.filePathAndNameLength + 4];
        infoData.fileSizeParameterLength = data[infoData.filePathAndNameLength + 5];
        // sid + optMode + filePathAndNameLengthSize + NameSize + dataFormatIdentifierSize + fileSizeParameterLengthSize + fileSizeUnCompressed + fileSizeCompressed
        infoData.totalSize = 1 + 1 + 2 + infoData.filePathAndNameLength  + 1 + 1 + infoData.fileSizeParameterLength + infoData.fileSizeParameterLength;
        for (int i = 0; i < infoData.fileSizeParameterLength; ++i) {
            infoData.fileSizeUnCompressed.emplace_back(data[i + infoData.filePathAndNameLength + 6]);
            infoData.fileSizeCompressed.emplace_back(data[i + infoData.filePathAndNameLength + infoData.fileSizeParameterLength + 6]);
        }

        for (int i = 0; i < infoData.fileSizeParameterLength; ++i) {
            infoData.fileSize = (i == 0) ? data[i + infoData.filePathAndNameLength + 6] : ((infoData.fileSize << 8) | data[i + infoData.filePathAndNameLength + 6]);
        }
    }
    else if (infoData.opt_mode == ModeOfOperation_DelFile) {
        GetPath(vec, infoData.filePathAndName, infoData.filePath);
        infoData.dataFormatIdentifier = 0xFF;
        infoData.fileSizeParameterLength = 0x00;
        // sid + optMode + filePathAndNameLengthSize + NameSize
        infoData.totalSize = 1 + 1 + 2 + infoData.filePathAndNameLength;
        infoData.fileSize = 0;
    }
    else if (infoData.opt_mode == ModeOfOperation_ReadFile) {
        GetPath(vec, infoData.filePathAndName, infoData.filePath);
        infoData.dataFormatIdentifier = data[4 + infoData.filePathAndNameLength];
        infoData.fileSizeParameterLength = 0x00;
        // sid + optMode + filePathAndNameLengthSize + NameSize + dataFormatIdentifierSize
        infoData.totalSize = 1 + 1 + 2 + infoData.filePathAndNameLength  + 1;
        infoData.fileSize = 0;
    }
    else if (infoData.opt_mode == ModeOfOperation_ReadDir) {
        infoData.filePathAndName = "";
        infoData.dataFormatIdentifier = 0xFF;
        infoData.fileSizeParameterLength = 0;
        // sid + optMode + filePathAndNameLengthSize + NameSize
        infoData.totalSize = 1 + 1 + 2 + infoData.filePathAndNameLength;
        infoData.fileSize = 0;
        for (uint32_t i = 0; i < vec.size(); ++i) {
            infoData.filePath += vec[i];
        }
    }
}

void
DiagServerUdsSid38::GetPath(std::vector<uint8_t> data, std::string& filePathAndName, std::string& filePath)
{
    DG_DEBUG << "DiagServerUdsSid38::GetPath";
    if (data.empty()) {
        DG_ERROR << "DiagServerUdsSid38::GetPath data is empty!";
        return;
    }

    std::string str1;
    for (uint32_t i = 0; i < data.size(); ++i) {
        str1 += data[i];
    }

    // str1 += '\0';
    filePathAndName = str1;
    reverse(data.begin(), data.end());
    std::vector<uint8_t> vec;
    bool flag = false;

    for (uint32_t j = 0; j < data.size(); ++j) {
        if ((data[j] == 0x2f) && (!flag)) {
            flag = true;
            ++j;
        }

        if (flag) {
            vec.emplace_back(data[j]);
        }
    }

    reverse(vec.begin(), vec.end());
    std::string str2;
    for (uint32_t i = 0; i < vec.size(); ++i) {
        str2 += vec[i];
    }

    // str2 += '\0';
    filePath = str2;
    return;
}

void
DiagServerUdsSid38::RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage)
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
    DiagServerTransPortCM::getInstance()->DiagMethodSend(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_FILE_TRANSFER, sub_func, service, udsData);

    // response
    for (auto item : udsData) {
        responseMessage.udsData.emplace_back(item);
    }

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}

void
DiagServerUdsSid38::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid38::PositiveResponse38";
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsSid38::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsSid38::NegativeResponse";
    DiagServerUdsMessage responseMessage;
    responseMessage.id = udsMessage.id;
    responseMessage.pendingRsp = udsMessage.pendingRsp;
    responseMessage.suppressPosRspMsgIndBit = udsMessage.suppressPosRspMsgIndBit;
    responseMessage.udsSa = udsMessage.udsSa;
    responseMessage.udsTa = udsMessage.udsTa;
    responseMessage.taType = udsMessage.taType;
    responseMessage.udsData.push_back(static_cast<uint8_t>(DiagServerNrcErrc::kNegativeHead));
    responseMessage.udsData.push_back(static_cast<uint8_t>(DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_FILE_TRANSFER));
    responseMessage.udsData.push_back(static_cast<uint8_t>(nrc_errc_));
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(responseMessage);
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
