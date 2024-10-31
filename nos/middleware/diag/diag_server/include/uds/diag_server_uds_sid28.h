/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid28.h is designed for diagnostic Communication Control.
 */

#ifndef DIAG_SERVER_UDS_SID28_H
#define DIAG_SERVER_UDS_SID28_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid28 : public DiagServerUdsBase {
public:
    DiagServerUdsSid28();
    virtual ~DiagServerUdsSid28();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

    bool EnableRxAndTx();
    bool EnableRxAndDisableTx();
    bool DisableRxAndEnableTx();
    bool DisableRxAndTx();
    bool EnableRxAndDisableTxWithAddress();
    bool EnableRxAndTxWithAddress();

private:
    DiagServerUdsSid28(const DiagServerUdsSid28 &);
    DiagServerUdsSid28 & operator = (const DiagServerUdsSid28 &);

private:
    static std::mutex mtx_;
    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID28_H
