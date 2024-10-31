/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid19.h is designed for diagnostic Read DTC Information.
 */

#ifndef DIAG_SERVER_UDS_SID19_H
#define DIAG_SERVER_UDS_SID19_H

#include <unordered_map>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid19 : public DiagServerUdsBase {
public:
    DiagServerUdsSid19();
    virtual ~DiagServerUdsSid19();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);
    void sendNegative(const DiagServerNrcErrc eNrc);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid19(const DiagServerUdsSid19 &);
    DiagServerUdsSid19 & operator = (const DiagServerUdsSid19 &);

private:
    DiagServerUdsMessage m_udsMessage;
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID19_H
