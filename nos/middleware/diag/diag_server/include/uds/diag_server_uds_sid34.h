/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid34.h is designed for diagnostic Request for download.
 */

#ifndef DIAG_SERVER_UDS_SID34_H
#define DIAG_SERVER_UDS_SID34_H

#include <vector>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid34 : public DiagServerUdsBase {
public:
    DiagServerUdsSid34();
    virtual ~DiagServerUdsSid34();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid34(const DiagServerUdsSid34 &);
    DiagServerUdsSid34 & operator = (const DiagServerUdsSid34 &);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID34_H
