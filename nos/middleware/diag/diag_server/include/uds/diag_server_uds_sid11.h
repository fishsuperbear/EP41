/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid11.h is designed for diagnostic Ecu Reset.
 */

#ifndef DIAG_SERVER_UDS_SID11_H
#define DIAG_SERVER_UDS_SID11_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid11 : public DiagServerUdsBase {
public:
    DiagServerUdsSid11();
    virtual ~DiagServerUdsSid11();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid11(const DiagServerUdsSid11 &);
    DiagServerUdsSid11 & operator = (const DiagServerUdsSid11 &);

private:
    static std::mutex mtx_;
    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID11_H
