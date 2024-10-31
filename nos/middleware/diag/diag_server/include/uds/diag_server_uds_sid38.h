/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid38.h is designed for diagnostic Request FileT ransfer Replace File.
 */

#ifndef DIAG_SERVER_UDS_SID38_H
#define DIAG_SERVER_UDS_SID38_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdio>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"
#include "diag/diag_server/include/common/diag_server_def.h"



namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid38 : public DiagServerUdsBase {
public:
    enum ModeOfOperation {
        ModeOfOperation_Reserve                               = 0x00,
        ModeOfOperation_AddFile                               = 0x01,
        ModeOfOperation_DelFile                               = 0x02,
        ModeOfOperation_ReplaceFile                           = 0x03,
        ModeOfOperation_ReadFile                              = 0x04,
        ModeOfOperation_ReadDir                               = 0x05,
    };

    struct DiagSid38ParameterInfo {
        uint8_t opt_mode;
        uint16_t filePathAndNameLength;
        std::string filePathAndName;
        std::string filePath;
        uint8_t dataFormatIdentifier;
        uint8_t fileSizeParameterLength;
        std::vector<uint8_t> fileSizeUnCompressed;
        std::vector<uint8_t> fileSizeCompressed;
        uint64_t fileSize;
        size_t totalSize;
    };

    DiagServerUdsSid38();
    virtual ~DiagServerUdsSid38();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);
    bool DealwithRequest(std::vector<uint8_t>& udsData);

    bool CheckTotalLength(const std::vector<uint8_t>& udsData);
    bool CheckOutOfRange(const std::vector<uint8_t>& udsData);
    bool CheckPath(const uint8_t configId, const std::string& filePath);
    void GetParameter(std::vector<uint8_t> data, DiagSid38ParameterInfo& udsData);
    void GetPath(std::vector<uint8_t> data, std::string& filePathAndName, std::string& filePath);

private:
    DiagServerUdsSid38(const DiagServerUdsSid38 &);
    DiagServerUdsSid38 & operator = (const DiagServerUdsSid38 &);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID38_H
