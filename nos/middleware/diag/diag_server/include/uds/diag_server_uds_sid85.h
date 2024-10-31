/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid85.h is designed for diagnostic Control DTC Setting.
 */

#ifndef DIAG_SERVER_UDS_SID85_H
#define DIAG_SERVER_UDS_SID85_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid85 : public DiagServerUdsBase {
public:
    DiagServerUdsSid85();
    virtual ~DiagServerUdsSid85();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid85(const DiagServerUdsSid85 &);
    DiagServerUdsSid85 & operator = (const DiagServerUdsSid85 &);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID85_H
