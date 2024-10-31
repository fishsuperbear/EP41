/*
 * Copyright (c) Hozon Auto Co., Ltd. 2022-2022. All rights reserved.
 * Description: fault lock
 */

#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <regex>
#include <ctime>
#include <time.h>
// #include <sys_ctr.h>
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/time_manager.h"
#include "phm_server/include/fault_lock/phm_fault_lock.h"
#include "phm_server/include/common/phm_server_persistency.h"
#include "phm_server/include/fault_lock/phm_fault_lock_signal_manager.h"

namespace hozon {
namespace netaos {
namespace phm_server {

// Lock fault record path.
const std::string FAULT_LOCK_LIST_FILE = "hz_fault_lock_list.json";

const std::string LOCK_FAULT_DATA = "LockFaultData";
const std::string FAULT_DATA_TO_HMI = "FaultDataToHMI";
const std::string LOCK_FAULT_NUMBER = "LockFaultNumber";

const unsigned int REPORT_CYCLE_TIME = 300;   // 300ms
const unsigned int RECORD_CYCLE_TIME = 1000;  // 1s

std::mutex FaultLock::mtx_;
FaultLock* FaultLock::m_ptrInstance = nullptr;

FaultLock::FaultLock()
: reportTimerFd(-1)
, recordTimerFd(-1)
, faultDataToHMI(NORMAL_FAULT_LOCK_VALUE)
, lockFaultData(NORMAL_FAULT_LOCK_VALUE)
, lockFaultNumber(0)
, m_threadPool(new ThreadPool(10))
, m_timerManager(new TimerManager())
, m_phmInterfaceFault2mcu(nullptr)
{
#ifdef BUILD_FOR_ORIN
    m_phmInterfaceFault2mcu = std::make_shared<PhmInterfaceFault2mcu>();
#endif
}

FaultLock*
FaultLock::getInstance()
{
    if (nullptr == m_ptrInstance) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == m_ptrInstance) {
            m_ptrInstance = new FaultLock();
        }
    }

    return m_ptrInstance;
}

void
FaultLock::Init()
{
    PHMS_INFO << "FaultLock::Init.";
#ifdef BUILD_FOR_ORIN
    if (nullptr != m_phmInterfaceFault2mcu) {
        m_phmInterfaceFault2mcu->Init();
    }
#endif
    // timer manager init
    if (nullptr != m_timerManager) {
        m_timerManager->Init();
    }

    FaultLockSignalManager::getInstance()->Init();
    PHMServerPersistency::getInstance()->GetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_NUMBER, this->lockFaultNumber);
    PHMServerPersistency::getInstance()->GetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_DATA, this->lockFaultData);
    this->faultDataToHMI = this->lockFaultData;

    RecordFaultData(LOCK_FAULT_NUMBER, this->lockFaultNumber);
    RecordFaultData(LOCK_FAULT_DATA, this->lockFaultData);
    RecordFaultData(FAULT_DATA_TO_HMI, this->faultDataToHMI);

    StartReportAndRecord();
    SyncLockFaultStatus();
    RecoverSelfRecoveryFault();
    RecoverInhibitWindowFault();
}

void
FaultLock::DeInit()
{
    PHMS_INFO << "FaultLock::DeInit.";
    StopReportAndRecord();
    FaultLockSignalManager::getInstance()->DeInit();
    if (nullptr != m_timerManager) {
        m_timerManager->DeInit();
        m_timerManager = nullptr;
    }

    if (nullptr != m_threadPool) {
        m_threadPool->Stop();
        m_threadPool = nullptr;
    }

#ifdef BUILD_FOR_ORIN
    if (nullptr != m_phmInterfaceFault2mcu) {
        m_phmInterfaceFault2mcu->DeInit();
        m_phmInterfaceFault2mcu = nullptr;
    }
#endif

    if (nullptr != m_ptrInstance) {
        delete m_ptrInstance;
        m_ptrInstance = nullptr;
    }
}

void
FaultLock::SetFaultDataToHMI(const uint64_t faultData) {
    PHMS_DEBUG << "FaultLock::SetFaultDataToHMI faultData: " << faultData;
    std::lock_guard<std::mutex> lck(mtx_);
    if (faultData == this->faultDataToHMI) {
        return;
    }

    this->faultDataToHMI = faultData;
    RecordFaultData(FAULT_DATA_TO_HMI, this->faultDataToHMI);
    PHMS_INFO << "FaultLock::SetFaultDataToHMI this->faultDataToHMI: " << this->faultDataToHMI;
}

void
FaultLock::SetLockFaultData(const uint64_t lockFaultData) {
    PHMS_DEBUG << "FaultLock::SetLockFaultData lockFaultData: " << lockFaultData;
    std::lock_guard<std::mutex> lck(mtx_);
    this->lockFaultData = lockFaultData;
    RecordFaultData(LOCK_FAULT_DATA, this->lockFaultData);
}

void
FaultLock::ReportFault(const FaultLockReportInfo& faultData)
{
    PHMS_DEBUG << "FaultLock::ReportFault.";

    FaultLockInfo faultLockInfo;
    uint32_t key = faultData.faultId * 100 + faultData.faultObj;
    bool bResult = PHMServerConfig::getInstance()->GetLockFaultInfo(key, faultLockInfo);
    // Determine whether this fault is in the lock fault table.
    if (!bResult) {
        PHMS_DEBUG << "FaultLock::ReportFault  fault: " << key << " is not lock fault.";
        return;
    }

    if(faultData.faultStatus) {
        if(faultLockInfo.isHandled) {
            PHMS_DEBUG << "FaultLock::ReportFault fault: " << key << " Current cycle is already handled.";
            return;
        }
    }

    FaultLockReportInfo* faultInfo = new FaultLockReportInfo;
    faultInfo->faultId = faultData.faultId;
    faultInfo->faultObj = faultData.faultObj;
    faultInfo->faultStatus = faultData.faultStatus;
    faultInfo->isInhibitWindowFault = faultData.isInhibitWindowFault;
    if (nullptr != m_threadPool) {
        m_threadPool->Commit(std::bind(&FaultLock::DealWithFaultTask, this, faultInfo));
    }
}

void
FaultLock::StartReportAndRecord()
{
    PHMS_INFO << "FaultLock::StartReportAndRecord faultDataToHMI: " << faultDataToHMI;
    if (nullptr != m_threadPool) {
        m_threadPool->Commit(std::bind(&FaultLock::ReportTask, this));
        m_threadPool->Commit(std::bind(&FaultLock::RecordTask, this));
    }
}

void
FaultLock::StopReportAndRecord()
{
    PHMS_INFO << "FaultLock::StopReportFaultDataToHMI.";
    PHMServerPersistency::getInstance()->SyncKeyValueFile(FAULT_LOCK_LIST_FILE);
    if (nullptr != m_timerManager) {
        m_timerManager->StopFdTimer(this->reportTimerFd);
        m_timerManager->StopFdTimer(this->recordTimerFd);
    }
}

void
FaultLock::SyncLockFaultStatus()
{
    PHMS_INFO << "FaultLock::SyncLockFaultStatus.";
    std::vector<std::string> vecKey;
    PHMServerPersistency::getInstance()->GetAllKeys(FAULT_LOCK_LIST_FILE, vecKey);
    if (vecKey.size() <= 4) {
        return;
    }

    std::map<std::string, bool> keyCheckMap
    {
        {"LockFaultSignal", false},
        {"LockFaultNumber", false},
        {"LockFaultData", false},
        {"FaultDataToHMI", false}
    };

    for (int index = 0; index < 4; ++index) {
        keyCheckMap[vecKey[index]] = true;
    }

    for (auto& keyCheckResult : keyCheckMap) {
        if (false == keyCheckResult.second) {
            PHMS_ERROR << "FaultLock::SyncLockFaultStatus hz_fault_lock_list.json format invalid";
            return;
        }
    }

    std::string str = "";
    uint32_t fault = 0;
    uint32_t lockedNumber = 0;
    for (uint i = 4; i < vecKey.size(); i++) {
        PHMServerPersistency::getInstance()->GetValue(FAULT_LOCK_LIST_FILE, vecKey[i], str);
        auto vec = Split(vecKey[i]);
        if (vec.size() < 4) {
            continue;
        }
        fault = strtoull(vec[1].c_str(), 0, 10) * 100 + strtoull(vec[3].c_str(), 0, 10);
        vec.clear();

        vec = Split(str);
        if (vec.size() < 8) {
            continue;
        }
        lockedNumber = strtoull(vec[7].c_str(), 0, 10);
        PHMS_INFO << "FaultLock::SyncLockFaultStatus fault: " << fault << " lockedNumber: " << lockedNumber;
        PHMServerConfig::getInstance()->SetLockFaultInfo(fault, lockedNumber);
    }
}

void
FaultLock::RecoverSelfRecoveryFault() {
    PHMS_INFO << "FaultLock::RecoverSelfRecoveryFault.";
    std::vector<FaultLockInfo> faultLockInfos;
    PHMServerConfig::getInstance()->GetLockFaultInfos(faultLockInfos);
    if(faultLockInfos.size() <= 0) {
        PHMS_ERROR << "FaultLock::RecoverSelfRecoveryFault faultLockInfos data error.";
        return;
    }

    FaultLockReportInfo faultLockReportInfo;
    faultLockReportInfo.faultId = 0;
    faultLockReportInfo.faultObj = 0;
    faultLockReportInfo.faultStatus = 0;
    faultLockReportInfo.isInhibitWindowFault = false;
    for (auto& faultLockInfo : faultLockInfos) {
        if (1 == faultLockInfo.isNeedToRecover) {
            faultLockReportInfo.faultId = faultLockInfo.faultId;
            faultLockReportInfo.faultObj = faultLockInfo.faultObj;
            ReportFault(faultLockReportInfo);
        }
    }
}

void
FaultLock::RecoverInhibitWindowFault() {
    PHMS_INFO << "FaultLock::RecoverInhibitWindowFault.";
    std::vector<std::string> vecKey;
    PHMServerPersistency::getInstance()->GetAllKeys(FAULT_LOCK_LIST_FILE, vecKey);
    if (vecKey.size() <= 4) {
        return;
    }

    std::string str = "";
    FaultLockReportInfo faultLockReportInfo;
    faultLockReportInfo.faultId = 0;
    faultLockReportInfo.faultObj = 0;
    faultLockReportInfo.faultStatus = 0;
    faultLockReportInfo.isInhibitWindowFault = true;
    for (uint i = 4; i < vecKey.size(); i++) {
        PHMServerPersistency::getInstance()->GetValue(FAULT_LOCK_LIST_FILE, vecKey[i], str);
        if (std::string::npos != str.find("IsInhibitWindowFault: 1", 0)) {
            auto vec = Split(vecKey[i], "\"");
            if (vec.size() < 1) {
                continue;
            }
            str = vec[0];
            vec.clear();

            vec = Split(str);
            if (vec.size() < 4) {
                continue;
            }
            faultLockReportInfo.faultId = strtoull(vec[1].c_str(), 0, 10);
            faultLockReportInfo.faultObj = strtoull(vec[3].c_str(), 0, 10);
            ReportFault(faultLockReportInfo);
        }
    }
}

void
FaultLock::ReportTask()
{
    PHMS_INFO << "FaultLock::ReportTask Run ";
    if (nullptr != m_timerManager) {
        m_timerManager->StartFdTimer(reportTimerFd, REPORT_CYCLE_TIME, std::bind(&FaultLock::ReportFaultDataToHMI, this, std::placeholders::_1), NULL, true);
    }
}

void
FaultLock::ReportFaultDataToHMI(void * data)
{
    PHMS_DEBUG << "FaultLock::ReportFaultDataToHMI faultDataToHMI: " << faultDataToHMI;
#ifdef BUILD_FOR_ORIN
    if (nullptr != m_phmInterfaceFault2mcu) {
        m_phmInterfaceFault2mcu->FaultToHMIToMCU(faultDataToHMI);
    }
#endif
}

void
FaultLock::RecordTask()
{
    PHMS_INFO << "FaultLock::RecordTask Run ";
    if (nullptr != m_timerManager) {
        m_timerManager->StartFdTimer(recordTimerFd, RECORD_CYCLE_TIME, std::bind(&FaultLock::RecordFaultDataToFile, this, std::placeholders::_1), NULL, true);
    }
}

void
FaultLock::RecordFaultDataToFile(void * data)
{
    PHMS_DEBUG << "FaultLock::RecordTask RecordFaultDataToFile ";
    PHMServerConfig::getInstance()->LockFaultDataToFile();
}

void
FaultLock::DealWithFaultTask(const FaultLockReportInfo* faultData)
{
    PHMS_DEBUG << "FaultLock::DealWithFaultTask Run start.";
    if (nullptr == faultData) {
        PHMS_ERROR << "FaultLock::DealWithFaultTask Run faultData is nullptr.";
        return;
    }

    uint32_t faultId = faultData->faultId;
    uint32_t faultObj = faultData->faultObj;
    uint32_t faultStatus = faultData->faultStatus;
    bool isInhibitWindowFault = faultData->isInhibitWindowFault;

    FaultLockInfo faultLockInfo;
    uint32_t key = faultId * 100 + faultObj;
    bool bResult = PHMServerConfig::getInstance()->GetLockFaultInfo(key, faultLockInfo);
    // Determine whether this fault is in the lock fault table.
    if (!bResult) {
        PHMS_WARN << "FaultLock::DealWithFaultTask Run this fault is not lock fault.";
        return;
    }

    uint32_t lockCount = faultLockInfo.lockCount;
    uint32_t recoverCount = faultLockInfo.recoverCount;
    std::string faultToHMIData = faultLockInfo.faultToHMIData;
    std::string lockFaultToHMIData = faultLockInfo.lockFaultToHMIData;
    uint32_t faultCount = faultLockInfo.faultCount;
    uint8_t isHandled = faultLockInfo.isHandled;
    uint32_t lockedNumber = faultLockInfo.lockedNumber;
    uint32_t faultRecoverCount = faultLockInfo.faultRecoverCount;
    uint8_t isNeedToRecover = faultLockInfo.isNeedToRecover;

    PHMS_DEBUG << "FaultLock::DealWithFaultTask Run before faultId: " << faultId \
                                                    << " faultObj: "<< faultObj \
                                                    << " lockCount: " << lockCount \
                                                    << " recoverCount: " << recoverCount \
                                                    << " faultToHMIData: " << faultToHMIData \
                                                    << " lockFaultToHMIData: " << lockFaultToHMIData \
                                                    << " faultCount: " << faultCount \
                                                    << " isHandled: " << isHandled \
                                                    << " lockedNumber: " << lockedNumber
                                                    << " faultRecoverCount: " << faultRecoverCount
                                                    << " isNeedToRecover: " << isNeedToRecover;

    // Corresponding treatment of fault generation and elimination.
    if (0 == faultStatus) {
        PHMS_DEBUG << "FaultLock::DealWithFaultTask Run Recover the fault.";
        FaultRecover(faultId, faultObj, faultStatus, isHandled, lockedNumber);
        faultCount = 0;
        isHandled = 0;
        lockedNumber = 0;
        faultRecoverCount = 0;
        isNeedToRecover = 0;
    }
    else if(1 == faultStatus) {
        PHMS_DEBUG << "FaultLock::DealWithFaultTask Run Report the fault.";
        // Judge whether the current fault cycle has been reported and generated.
        if (isHandled) {
            PHMS_DEBUG << "FaultLock::DealWithFaultTask Run Current cycle is already handled.";
            return;
        }

        isHandled = 1;
        faultCount++;
        if((PHMServerConfig::getInstance()->IsBlockedFault(faultId * 100 + faultObj)) || (faultCount < lockCount)) {
            if (!lockedNumber) {
                // Lock fault send to HMI.
                FaultLockSignalManager::getInstance()->SetFaultSignalNum((faultId * 100 + faultObj), faultStatus);
                uint64_t dataToHMI = FaultLockSignalManager::getInstance()->GetFaultData();
                PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecover dataToHMI: " << dataToHMI;
                SetFaultDataToHMI(dataToHMI);
            }
        }
        else {
            // 调用持久化接口写入锁存数据
            PHMS_DEBUG << "FaultLock::DealWithFaultTask Run  HW interface to record the fault.";
            lockedNumber ++;
            std::string strKey = "FaultId: " + std::to_string(faultId) + " FaultObj: " +  std::to_string(faultObj);
            std::string strValue = "FaultCount: " + std::to_string(faultCount) + " IsHandled: " +  std::to_string(isHandled) + " IsInhibitWindowFault: " + std::to_string(isInhibitWindowFault)
                                                + " LockedNumber: " +  std::to_string(lockedNumber) + " LockedManagementPlaneTime: " +  GetCurrentManagementPlaneTime() + " LockedDataPlaneTime: " +  GetCurrentDataPlaneTime();
            bool bRecordFault = RecordLockFault(strKey, strValue);
            if (bRecordFault) {
                PHMS_DEBUG << "FaultLock::DealWithFaultTask Run  Lock the fault write success.";
                FaultRecord(faultId, faultObj, faultStatus, isHandled, lockedNumber);
                faultCount = 0;
            }
            else {
                PHMS_WARN << "FaultLock::DealWithFaultTask Run  Lock the fault write failed.";
                lockedNumber --;
            }
        }

        faultRecoverCount = 0;
        isNeedToRecover = 0;
    }

    PHMS_DEBUG << "FaultLock::DealWithFaultTask Run after faultId: " << faultId \
                                                    << " faultObj: "<< faultObj \
                                                    <<  " lockCount: " << lockCount \
                                                    <<  " recoverCount: " << recoverCount \
                                                    <<  " faultToHMIData: " << faultToHMIData \
                                                    << " lockFaultToHMIData: " << lockFaultToHMIData \
                                                    <<  " faultCount: " << faultCount \
                                                    <<  " isHandled: " << isHandled \
                                                    <<  " lockedNumber: " << lockedNumber
                                                    <<  " faultRecoverCount: " << faultRecoverCount
                                                    <<  " isNeedToRecover: " << isNeedToRecover;

    faultLockInfo.faultCount = faultCount;
    faultLockInfo.isHandled = isHandled;
    faultLockInfo.lockedNumber = lockedNumber;
    faultLockInfo.faultRecoverCount = faultRecoverCount;
    faultLockInfo.isNeedToRecover = isNeedToRecover;
    PHMServerConfig::getInstance()->SetLockFaultInfo(key, faultLockInfo);
}

void
FaultLock::FaultRecover(const uint32_t faultId, const uint32_t faultObj, const uint32_t faultStatus, const uint32_t isHandled, const uint32_t lockedNumber)
{
    PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecover faultId: " << faultId << " faultObj: " << faultObj;
    std::string strKey = "FaultId: " + std::to_string(faultId) + " FaultObj: " +  std::to_string(faultObj);
    if (lockedNumber) {
        RemoveLockFault(strKey);
        FaultLockSignalManager::getInstance()->SetFaultSignalNum((faultId * 100 + faultObj), faultStatus, lockedNumber, true);
        uint64_t lockFaultData = FaultLockSignalManager::getInstance()->GetFaultData(true);
        PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecover lockFaultData: " << lockFaultData;
        SetLockFaultData(lockFaultData);
    }

    if (isHandled) {
        FaultLockSignalManager::getInstance()->SetFaultSignalNum((faultId * 100 + faultObj), faultStatus);
    }

    uint64_t dataToHMI = FaultLockSignalManager::getInstance()->GetFaultData();
    PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecover dataToHMI: " << dataToHMI;
    SetFaultDataToHMI(dataToHMI);
}

void
FaultLock::FaultRecord(const uint32_t faultId, const uint32_t faultObj, const uint32_t faultStatus, const uint32_t isHandled, const uint32_t lockedNumber)
{
    PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecord faultId: " << faultId << " faultObj: " << faultObj;
    FaultLockSignalManager::getInstance()->SetFaultSignalNum((faultId * 100 + faultObj), faultStatus, lockedNumber, true);
    uint64_t lockFaultData = FaultLockSignalManager::getInstance()->GetFaultData(true);
    PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecord lockFaultData: " << lockFaultData;
    SetLockFaultData(lockFaultData);

    if (isHandled) {
        FaultLockSignalManager::getInstance()->SetFaultSignalNum((faultId * 100 + faultObj), faultStatus);
    }

    uint64_t dataToHMI = FaultLockSignalManager::getInstance()->GetFaultData();
    PHMS_DEBUG << "FaultLock::DealWithFaultTask FaultRecord dataToHMI: " << dataToHMI;
    SetFaultDataToHMI(dataToHMI);
}

bool
FaultLock::RecordLockFault(const std::string& key, const std::string& value) {
    PHMS_INFO << "FaultLock::RecordLockFault key: " << key << " value: " << value;
    if (!(PHMServerPersistency::getInstance()->IsHasKey(FAULT_LOCK_LIST_FILE, key))) {
        this->lockFaultNumber++;
        PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_NUMBER, this->lockFaultNumber);
    }

    if (!(PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, key, value))) {
        PHMS_WARN << "FaultLock::RecordLockFault SetValue failed.";
        return false;
    }

    if (!(PHMServerPersistency::getInstance()->SyncKeyValueFile(FAULT_LOCK_LIST_FILE))) {
        PHMS_ERROR << "FaultLock::RecordLockFault SyncToStorage failed.";
        return false;
    }

    return true;
}

bool
FaultLock::RecordFaultData(const std::string& key, const uint64_t value) {
    PHMS_INFO << "FaultLock::RecordFaultData key: " << key << " value: " << value;
    if (!(PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, key, value))) {
        PHMS_WARN << "FaultLock::RecordFaultData SetValue failed.";
        return false;
    }

    if (!(PHMServerPersistency::getInstance()->SyncKeyValueFile(FAULT_LOCK_LIST_FILE))) {
        PHMS_ERROR << "FaultLock::RecordFaultData SyncToStorage failed.";
        return false;
    }

    return true;
}

bool
FaultLock::RemoveLockFault(const std::string& key) {
    PHMS_INFO << "FaultLock::RemoveLockFault key: " << key;
    if (!(PHMServerPersistency::getInstance()->IsHasKey(FAULT_LOCK_LIST_FILE, key))) {
        PHMS_WARN << "FaultLock::RemoveLockFault not have this key.";
        return false;
    }

    // Remove key.
    if (!(PHMServerPersistency::getInstance()->RemoveKey(FAULT_LOCK_LIST_FILE, key))) {
        PHMS_WARN << "FaultLock::RemoveLockFault RemoveKey failed.";
        return false;
    }

    this->lockFaultNumber--;
    PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_NUMBER, this->lockFaultNumber);

    if (!(PHMServerPersistency::getInstance()->SyncKeyValueFile(FAULT_LOCK_LIST_FILE))) {
        PHMS_WARN << "FaultLock::RecordFaultData SyncToStorage failed.";
        return false;
    }

    return true;
}

std::vector<std::string>
FaultLock::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

std::string
FaultLock::GetCurrentDataPlaneTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return std::string(tmp);
}

std::string
FaultLock::GetCurrentManagementPlaneTime()
{
    // struct timespec ts;
    // clock_gettime(CLOCK_VIRTUAL, &ts);
    // char tmp[64] = {0};
    // strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", gmtime(&ts.tv_sec));
    // return std::string(tmp);
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return std::string(tmp);
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon