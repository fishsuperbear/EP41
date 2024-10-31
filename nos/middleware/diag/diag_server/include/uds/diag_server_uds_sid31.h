/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid31.h is designed for diagnostic Routine Control.
 */

#ifndef DIAG_SERVER_UDS_SID31_H
#define DIAG_SERVER_UDS_SID31_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid31 : public DiagServerUdsBase {

public:
    enum Sid31RidStatus {
        Sid31RidStatus_Default,
        Sid31RidStatus_Started,
        Sid31RidStatus_Stopped
    };

    enum Sid31SubFuncId {
        Sid31SubFuncId_Start        = 0x01,
        Sid31SubFuncId_Stop         = 0x02,
        Sid31SubFuncId_Result       = 0x03,
    };

    DiagServerUdsSid31();
    virtual ~DiagServerUdsSid31();

    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:

    void InitRidStatusMap();

    bool CheckActiveSession(const uint16_t rid);
    bool CheckSecurity(const uint16_t rid);
    bool CheckRid(const uint16_t rid);
    bool CheckSubFunction(const uint16_t rid, const uint8_t sub_func_id);
    bool CheckTotalLength(const uint16_t rid, const uint8_t sub_func_id, const size_t length, const bool isReply = false);
    bool CheckSequence(const uint16_t rid, bool bStart = false);
    void UpdateRidStatus(const uint16_t rid, const Sid31RidStatus& status);
    void RequestToExternalService(const std::vector<std::string> service, const DiagServerUdsMessage& udsMessage);

    bool Start(const uint16_t rid, std::vector<uint8_t>& udsData);
    void StartTest(std::vector<uint8_t>& udsData);
    void StartCheckProgrammingPreconditions(std::vector<uint8_t>& udsData);
    void StartInstallation(std::vector<uint8_t>& udsData);
    void StartSwitchRunningPartition(std::vector<uint8_t>& udsData);
    void StartCheckProgrammingDependencies(std::vector<uint8_t>& udsData);
    void StartReportFaultOccur(std::vector<uint8_t>& udsData);
    void StartReportFaultRecover(std::vector<uint8_t>& udsData);
    void StartRefreshFaultFile(std::vector<uint8_t>& udsData);

    bool Stop(const uint16_t rid, std::vector<uint8_t>& udsData);
    void StopTest(std::vector<uint8_t>& udsData);
    void StopReportFaultOccur(std::vector<uint8_t>& udsData);
    void StopReportFaultRecover(std::vector<uint8_t>& udsData);

    bool Result(const uint16_t rid, std::vector<uint8_t>& udsData);
    void ResultTest(std::vector<uint8_t>& udsData);
    void ResultQueryCurrentFault(std::vector<uint8_t>& udsData);
    void ResultQueryDtcByFault(std::vector<uint8_t>& udsData);
    void ResultQueryFaultByDtc(std::vector<uint8_t>& udsData);

private:
    DiagServerUdsSid31(const DiagServerUdsSid31 &);
    DiagServerUdsSid31 & operator = (const DiagServerUdsSid31 &);

private:
    static std::mutex mtx_;

    DiagServerNrcErrc nrc_errc_;
    std::unordered_map<uint16_t, Sid31RidStatus> rid_status_map_;
};
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID31_H
