/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid2E.h is designed for diagnostic Write DataByIdentifier.
 */

#ifndef DIAG_SERVER_UDS_SID2E_H
#define DIAG_SERVER_UDS_SID2E_H

#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid2E : public DiagServerUdsBase {
public:
    DiagServerUdsSid2E();
    virtual ~DiagServerUdsSid2E();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

    bool WriteDidData(const uint16_t did, const std::vector<uint8_t>& didData);
    bool WriteDidDataCheck(const uint16_t did, const std::vector<uint8_t>& didData);

private:
    DiagServerUdsSid2E(const DiagServerUdsSid2E &);
    DiagServerUdsSid2E & operator = (const DiagServerUdsSid2E &);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID2E_H
