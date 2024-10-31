/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid37.h is designed for diagnostic Request Transfer Exit.
 */

#ifndef DIAG_SERVER_UDS_SID37_H
#define DIAG_SERVER_UDS_SID37_H

#include <vector>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"



namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsMgr;

class DiagServerUdsSid37 : public DiagServerUdsBase {
public:
    DiagServerUdsSid37();
    virtual ~DiagServerUdsSid37();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid37(const DiagServerUdsSid37 &);
    DiagServerUdsSid37 & operator = (const DiagServerUdsSid37 &);

    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID37_H
