/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid36.h is designed for diagnostic Transfer data.
 */

#ifndef DIAG_SERVER_UDS_SID36_H
#define DIAG_SERVER_UDS_SID36_H

#include <vector>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"



namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsMgr;

class DiagServerUdsSid36 : public DiagServerUdsBase {
public:
    DiagServerUdsSid36();
    virtual ~DiagServerUdsSid36();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid36(const DiagServerUdsSid36 &);
    DiagServerUdsSid36 & operator = (const DiagServerUdsSid36 &);

    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
    std::vector<uint8_t> uds_data_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID36_H
