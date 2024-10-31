/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid14.h is designed for diagnostic Clear dtc.
 */

#ifndef DIAG_SERVER_UDS_SID14_H
#define DIAG_SERVER_UDS_SID14_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid14 : public DiagServerUdsBase {
public:
    DiagServerUdsSid14();
    virtual ~DiagServerUdsSid14();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid14(const DiagServerUdsSid14 &);
    DiagServerUdsSid14 & operator = (const DiagServerUdsSid14 &);

private:
    static std::mutex mtx_;
    DiagServerUdsMessage m_udsMessage;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID14_H
