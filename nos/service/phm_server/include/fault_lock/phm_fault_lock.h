/*
 * Copyright (c) Hozon Auto Co., Ltd. 2022-2022. All rights reserved.
 * Description: fault lock
 */

#ifndef PHM_FAULT_LOCK_H
#define PHM_FAULT_LOCK_H

#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/thread_pool.h"
#include "phm_server/include/common/time_manager.h"
#include "phm_server/include/fault_manager/serviceInterface/phm_interface_fault2mcu.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class FaultLock {

public:
    static FaultLock* getInstance();
    void Init();
    void DeInit();

    void ReportFault(const FaultLockReportInfo& reportInfo);

private:
    void StartReportAndRecord();
    void StopReportAndRecord();
    void SyncLockFaultStatus();
    void RecoverSelfRecoveryFault();
    void RecoverInhibitWindowFault();

    void ReportTask();
    void ReportFaultDataToHMI(void * data);
    void RecordTask();
    void RecordFaultDataToFile(void * data);
    void DealWithFaultTask(const FaultLockReportInfo* faultData);
    void FaultRecover(const uint32_t faultId, const uint32_t faultObj, const uint32_t faultStatus, const uint32_t isHandled, const uint32_t lockedNumber);
    void FaultRecord(const uint32_t faultId, const uint32_t faultObj, const uint32_t faultStatus, const uint32_t isHandled, const uint32_t lockedNumber);
    void SetFaultDataToHMI(const uint64_t faultData);
    void SetLockFaultData(const uint64_t lockFaultData);

    bool RecordLockFault(const std::string& key, const std::string& value);
    bool RemoveLockFault(const std::string& key);
    bool RecordFaultData(const std::string& key, const uint64_t value);

    std::string GetCurrentDataPlaneTime();
    std::string GetCurrentManagementPlaneTime();
    std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = " ");

private:
    FaultLock();
    FaultLock(const FaultLock &);
    FaultLock & operator = (const FaultLock &);

private:
    static FaultLock* m_ptrInstance;
    int reportTimerFd;
    int recordTimerFd;
    uint64_t faultDataToHMI;
    uint64_t lockFaultData;
    uint64_t lockFaultNumber;
    static std::mutex mtx_;

    std::shared_ptr<ThreadPool> m_threadPool;
    std::shared_ptr<TimerManager> m_timerManager;
    std::shared_ptr<PhmInterfaceFault2mcu> m_phmInterfaceFault2mcu;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_LOCK_H