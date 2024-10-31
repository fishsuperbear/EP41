/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid22.h is designed for diagnostic Read Data By Identifier.
 */

#ifndef DIAG_SERVER_UDS_SID22_H
#define DIAG_SERVER_UDS_SID22_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid22 : public DiagServerUdsBase {

public:
    DiagServerUdsSid22();
    virtual ~DiagServerUdsSid22();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);
    bool DidSupportAndSecurityCheck(const uint16_t did);
    bool ReadDidData(const uint16_t did, std::vector<uint8_t>& udsData);
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid22(const DiagServerUdsSid22 &);
    DiagServerUdsSid22 & operator = (const DiagServerUdsSid22 &);

    static std::mutex mtx_;
    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID22_H
