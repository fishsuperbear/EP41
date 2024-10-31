/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault manager client
*/

#include <ctime>
#include <time.h>
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/timer_manager.h"
#include "phm/fault_manager/include/fault_auto_recovery.h"
#include "phm/common/include/phm_config.h"


namespace hozon {
namespace netaos {
namespace phm {


FaultAutoRecovery::FaultAutoRecovery(std::function<void(const SendFault_t&)> cb)
: m_faultReporterCb(cb)
{
}

FaultAutoRecovery::~FaultAutoRecovery()
{
}

void
FaultAutoRecovery::Init()
{
    PHM_INFO << "FaultAutoRecovery::Init";
    fault_map_.clear();
    fault_once_map_.clear();
}

void
FaultAutoRecovery::DeInit()
{
    PHM_INFO << "FaultAutoRecovery::DeInit";
    fault_map_.clear();
    fault_once_map_.clear();
}

bool
FaultAutoRecovery::NotifyFaultInfo(const SendFault_t& faultInfo)
{
    PHM_DEBUG << "FaultAutoRecovery::NotifyFaultInfo faultInfo.faultId: " << faultInfo.faultId
              << ", faultObj: " << (int)faultInfo.faultObj
              << ", faultStatus: " << (int)faultInfo.faultStatus
              << ", faultDebounce: " << static_cast<uint>(faultInfo.faultDebounce.debounceType)
              << ", debounceCount: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceCount)
              << ", debounceTime: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceTime)
              << ", isAutoRecovery: " << faultInfo.isAutoRecovery
              << ", autoRecoveryTime: " << faultInfo.autoRecoveryTime;

    if (!faultInfo.isAutoRecovery) {
        return DealWithFault(faultInfo);
    }

    if (!faultInfo.faultStatus) {
        PHM_ERROR << "FaultAutoRecovery::NotifyFaultInfo fault: " << faultInfo.faultId*100 + faultInfo.faultObj << ", faultStatus error.";
        return false;
    }

    uint32_t faultKey = faultInfo.faultId*100 + faultInfo.faultObj;
    auto itr_fault_find = fault_map_.find(faultKey);
    if (itr_fault_find != fault_map_.end()) {
        if (-1 != itr_fault_find->second) {
            StopTimer(faultKey);
        }
    }
    else {
        fault_map_.insert(std::make_pair(faultKey, -1));
        ReportFault(faultInfo);
    }

    StartTimer(faultKey, static_cast<unsigned int>(faultInfo.autoRecoveryTime));
    return true;
}

void
FaultAutoRecovery::StartTimer(const uint32_t faultKey, unsigned int msTime)
{
    PHM_INFO << "FaultAutoRecovery::StartTimer faultKey: " << faultKey << ", msTime: " << msTime;
    auto itr_fault_find = fault_map_.find(faultKey);
    if (itr_fault_find != fault_map_.end()) {
        TimerManager::Instance()->StartFdTimer(itr_fault_find->second, msTime, std::bind(&FaultAutoRecovery::RecoveryTimeoutCallback, this, std::placeholders::_1), const_cast<uint32_t*>(&faultKey), false);
    }
}

void
FaultAutoRecovery::StopTimer(const uint32_t faultKey)
{
    PHM_INFO << "FaultAutoRecovery::StopTimer faultKey: " << faultKey ;
    auto itr_fault_find = fault_map_.find(faultKey);
    if (itr_fault_find != fault_map_.end()) {
        TimerManager::Instance()->StopFdTimer(itr_fault_find->second);
    }
}

bool
FaultAutoRecovery::DealWithFault(const SendFault_t& faultInfo)
{
    uint32_t faultKey = faultInfo.faultId*100 + faultInfo.faultObj;
    auto itr_fault_once_find = fault_once_map_.find(faultKey);
    if (itr_fault_once_find != fault_once_map_.end()) {
        if (faultInfo.faultStatus != itr_fault_once_find->second) {
            itr_fault_once_find->second = faultInfo.faultStatus;
            return ReportFault(faultInfo);
        }
        PHM_DEBUG << "FaultAutoRecovery::DealWithFault repeat report, faultKey:" << faultKey;
    }
    else {
        fault_once_map_.insert(std::make_pair(faultKey, faultInfo.faultStatus));
        return ReportFault(faultInfo);
    }

    return true;
}

bool
FaultAutoRecovery::ReportFault(const SendFault_t& faultInfo)
{
    PHM_DEBUG << "FaultAutoRecovery::ReportFault faultInfo.faultId: " << faultInfo.faultId
              << ", faultObj: " << (int)faultInfo.faultObj
              << ", faultStatus: " << (int)faultInfo.faultStatus
              << ", faultDebounce: " << static_cast<uint>(faultInfo.faultDebounce.debounceType)
              << ", debounceCount: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceCount)
              << ", debounceTime: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceTime);

    if (nullptr == m_faultReporterCb) {
        return false;
    }

    m_faultReporterCb(faultInfo);
    return true;
}

void
FaultAutoRecovery::RecoveryTimeoutCallback(void* data)
{
    PHM_INFO << "FaultAutoRecovery::RecoveryTimeoutCallback";
    uint32_t* faultKey = (uint32_t*)data;
    SendFault_t sendFault(*faultKey / 100, *faultKey % 100, 0x00);
    ReportFault(sendFault);
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
