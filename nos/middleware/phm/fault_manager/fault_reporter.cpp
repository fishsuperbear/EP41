/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault manager client
*/

#include <vector>
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/timer_manager.h"
#include "phm/fault_manager/include/fault_dispatcher.h"
#include "phm/fault_manager/include/fault_auto_recovery.h"
#include "phm/fault_manager/include/fault_inhibit_judge.h"
#include "phm/fault_manager/include/fault_reporter.h"
#include "phm/fault_manager/include/fault_debounce_count.h"
#include "phm/fault_manager/include/fault_debounce_time.h"
#include "phm/common/include/phm_config.h"


namespace hozon {
namespace netaos {
namespace phm {

std::mutex FaultReporter::mtx_;


FaultReporter::FaultReporter()
{
    state_client_ = std::make_unique<hozon::netaos::sm::StateClient>();
    faultDispatcher_ = std::make_unique<FaultDispatcher>();
    m_spFaultAutoRecovery = std::make_shared<FaultAutoRecovery>(std::bind(&FaultReporter::FaultAutoRecoveryCallback, this, std::placeholders::_1));
}

FaultReporter::~FaultReporter()
{
}

int32_t
FaultReporter::Init(std::function<void(ReceiveFault_t)> fault_receive_callback, const std::string& processName)
{
    if (nullptr != state_client_) {
        state_client_->SetProcessName(processName);
    }

    int32_t res = 0;
    if (nullptr != faultDispatcher_) {
        res = faultDispatcher_->Init(fault_receive_callback);
    }

    if (m_spFaultAutoRecovery) {
        m_spFaultAutoRecovery->Init();
    }

    return res;
}

void
FaultReporter::DeInit()
{
    PHM_INFO << "FaultReporter::DeInit enter!";
    if (m_spFaultAutoRecovery) {
        m_spFaultAutoRecovery->DeInit();
        m_spFaultAutoRecovery = nullptr;
    }

    if (nullptr != faultDispatcher_) {
        faultDispatcher_->Deinit();
        faultDispatcher_ = nullptr;
    }

    // PHM_INFO << "FaultReporter::DeInit finish!";
}

int32_t
FaultReporter::ReportFault(const SendFault_t& faultInfo, std::shared_ptr<ModuleConfig> cfg)
{
    PHM_DEBUG << "FaultReporter::ReportFault faultInfo.faultId: " << faultInfo.faultId
              << " faultObj: " << (int)faultInfo.faultObj
              << " faultStatus: " << (int)faultInfo.faultStatus
              << " faultDebounce: " << static_cast<uint>(faultInfo.faultDebounce.debounceType)
              << " debounceCount: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceCount)
              << " debounceTime: " << static_cast<uint>(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceTime);

    // check fault exist
    PhmFaultInfo cPhmFaultInfo;
    bool res = PHMConfig::getInstance()->GetFaultInfoByFault(faultInfo.faultId * 100 + faultInfo.faultObj, cPhmFaultInfo);
    if (!res) {
        PHM_WARN << "FaultReporter::ReportFault unkown fault! faultKey: " << faultInfo.faultId * 100 + faultInfo.faultObj;
        return -1;
    }

    if (cfg->IsInhibitFault(faultInfo.faultId * 100 + faultInfo.faultObj)) {
        PHM_INFO << "FaultReporter::ReportFault inhibit faultId: " << faultInfo.faultId << " faultObj: " << (int)faultInfo.faultObj;
        return 0;
    }

    FaultInhibitJudge* pcFaultInhibitJudge = FaultInhibitJudge::getInstance();
    const uint8_t result = pcFaultInhibitJudge->CheckReportCondition();
    if (PHM_INHIBIT_TYPE_NONE != result) {
        static std::unordered_map<uint8_t, std::string> Info = {
            { PHM_INHIBIT_TYPE_OTA, "OTA"},
            { PHM_INHIBIT_TYPE_CALIBRATION, "CALIBRATION"},
            { PHM_INHIBIT_TYPE_PARKING, "PARKING"},
            { PHM_INHIBIT_TYPE_85, "85 OFF"},
            { PHM_INHIBIT_TYPE_POWERMODE_OFF, "POWERMODE_OFF"},
            { PHM_INHIBIT_TYPE_RUNNING_MODE, "RUNNING_MODE"}
        };
        PHM_INFO << "FaultReporter::ReportFault inhibit type:" << (int)result << ",msg:" << Info[result];
        return 0;
    }

    DebounceType cDebounceType = faultInfo.faultDebounce.debounceType;
    Fault* fault = GenFault(faultInfo.faultId, faultInfo.faultObj, cfg->GetModuleName());
    if (0 == faultInfo.faultDebounce.debounceSetting.countDebounce.debounceCount
        && 0 == faultInfo.faultDebounce.debounceSetting.countDebounce.debounceTime) {
        cDebounceType = DebounceType::DEBOUNCE_TYPE_UNUSE;
    }

    if (cDebounceType == DebounceType::DEBOUNCE_TYPE_COUNT) {
        fault->UseCountBaseDebouncePolicy(faultInfo.faultDebounce.debounceSetting.countDebounce.debounceCount,
                                          faultInfo.faultDebounce.debounceSetting.countDebounce.debounceTime);
    }
    else if (cDebounceType == DebounceType::DEBOUNCE_TYPE_TIME) {
        fault->UseTimeBaseDebouncePolicy(faultInfo.faultDebounce.debounceSetting.timeDebounce.debounceTime);
    }

    FaultStat_t faultStatus = fault->Report(faultInfo.faultStatus);
    if (faultStatus == IMMATURE) {
        return 0;
    }

    bool ret = m_spFaultAutoRecovery->NotifyFaultInfo(faultInfo);
    return (ret == true) ? 1 : 0;
}

bool
FaultReporter::ReportFaultImmediate(Fault* fault, const std::uint8_t faultStatus)
{
    std::lock_guard<std::mutex> lck(mtx_);
    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    PhmFaultInfo faultInfo;
    PHMConfig::getInstance()->GetFaultInfoByFault(fault->GetFaultId()*100+fault->GetFaultObj(), faultInfo);
    SendFaultPack_t pack;
    pack.faultDomain = faultInfo.faultProcess;
    pack.faultOccurTime = local_time;
    pack.faultId = fault->GetFaultId();
    pack.faultObj = fault->GetFaultObj();
    pack.faultStatus = faultStatus;

    if (nullptr != state_client_) {
        std::string processName = state_client_->GetProcessName();
        PHM_INFO << "FaultReporter::ReportFaultImmediate processName:" << processName;
        std::string::size_type idx = processName.find("Process");
        if (idx != std::string::npos) {
            pack.faultDomain = processName;
        }
    }

    if(IsPrefixMatch(fault->GetModuleName(), INNER_PREFIX)) {
        faultDispatcher_->PushFaultQueue(pack);
    }

    faultDispatcher_->Send(pack);
    return true;
}

bool FaultReporter::IsPrefixMatch(std::string& str, const std::string& prefix)
{
    if(str.substr(0, prefix.length()) == prefix) {
        return true;
    }
    return false;
}

Fault*
FaultReporter::GenFault(const std::uint32_t faultId, const std::uint8_t faultObj, std::string moduleName)
{
    if (0 == faultMap_.count(faultId * 100 + faultObj)) {
        PHM_DEBUG << "FaultReporter::GenFault faultId: " << faultId << " faultObj: " << (uint32_t)faultObj ;
        Fault* fault = new Fault(faultId, faultObj, moduleName, std::bind(&FaultReporter::ReportFaultImmediate, this, std::placeholders::_1, std::placeholders::_2));
        faultMap_[faultId * 100 + faultObj] = fault;
    }

    return faultMap_[faultId * 100 + faultObj];
}

void
FaultReporter::FaultAutoRecoveryCallback(const SendFault_t& faultInfo)
{
    Fault* fault = GenFault(faultInfo.faultId, faultInfo.faultObj, INNER_PREFIX + "_auto_recovery");
    ReportFaultImmediate(fault, faultInfo.faultStatus);
    return;
}

Fault::Fault(const std::uint32_t faultId, const std::uint8_t faultObj, std::string moduleName, std::function<bool(Fault*, const std::uint8_t)> reportFaultCb)
: faultId_(faultId)
, faultObj_(faultObj)
, moduleName_(moduleName)
, m_reportFaultCb(reportFaultCb)
{
    debouncePolicyPtr_ = std::make_shared<DebounceCount>(TimerManager::Instance(), 1, 0);
}

Fault::~Fault()
{
    debouncePolicyPtr_ = nullptr;
}

void
Fault::UseCountBaseDebouncePolicy(const std::uint32_t maxCount, const std::uint32_t timeoutMs)
{
    if (maxCount != debouncePolicyPtr_->GetDebouncePolicy().val[0]
        || timeoutMs != debouncePolicyPtr_->GetDebouncePolicy().val[1]) {
        // PHM_DEBUG << "Fault::UseCountBaseDebouncePolicy maxCount " << maxCount << " timeoutMs " << timeoutMs;
        debouncePolicyPtr_ = nullptr;
        debouncePolicyPtr_ = std::make_shared<DebounceCount>(TimerManager::Instance(), maxCount, timeoutMs);
    }
}

void
Fault::UseTimeBaseDebouncePolicy(const std::uint32_t timeoutMs)
{
    if (timeoutMs != debouncePolicyPtr_->GetDebouncePolicy().val[0]) {
        // PHM_DEBUG << "Fault::UseTimeBaseDebouncePolicy timeoutMs " << timeoutMs;
        debouncePolicyPtr_ = nullptr;
        debouncePolicyPtr_ = std::make_shared<DebounceTime>(TimerManager::Instance(), timeoutMs);
        std::shared_ptr<DebounceTime> timePolicyPtr = std::dynamic_pointer_cast<DebounceTime>(debouncePolicyPtr_);

        timePolicyPtr->RegistDebounceTimeoutCallback(std::bind(&Fault::TimeBaseDebouncepolicyTimeoutCallback, this));
    }
}

void
Fault::TimeBaseDebouncepolicyTimeoutCallback()
{
    // PHM_DEBUG << "HzFMClient::TimeDebounceTimeoutCallback";
    m_reportFaultCb(this, MATURE);
}

FaultStat_t
Fault::Report(const std::uint8_t faultStatus)
{
    PHM_DEBUG << "Fault::Report faultStatus: " << (uint32_t)faultStatus;
    if (debouncePolicyPtr_->GetDebouncePolicy().type == TYPE_COUNT) {
        std::shared_ptr<DebounceCount> countPolicyPtr = std::dynamic_pointer_cast<DebounceCount>(debouncePolicyPtr_);
        if (faultStatus == 0) {
            countPolicyPtr->Clear();
            countPolicyPtr->StopDebounceTimer();
            return RECOVER;
        }

        countPolicyPtr->Act();
        if (countPolicyPtr->isMature()) {
            PHM_DEBUG << "Fault::Report TYPE_COUNT isMature";
            countPolicyPtr->StopDebounceTimer();
            countPolicyPtr->Clear();
            return MATURE;
        }
        else {
            PHM_DEBUG << "Fault::Report TYPE_COUNT not isMature";
            countPolicyPtr->StartDebounceTimer();
        }
    }
    else {
        std::shared_ptr<DebounceTime> timePolicyPtr = std::dynamic_pointer_cast<DebounceTime>(debouncePolicyPtr_);
        if (faultStatus == 0) {
            timePolicyPtr->Clear();
            timePolicyPtr->StopDebounceTimer();
            return IMMATURE;
        }

        if (timePolicyPtr->isMature()) {
            PHM_DEBUG << "Fault::Report TYPE_TIME isMature";
            timePolicyPtr->Clear();
        }

        timePolicyPtr->StartDebounceTimer();
    }

    return IMMATURE;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
