#include "phm_server/include/fault_manager/analysis/fault_analysis.h"
#include "phm_server/include/fault_manager/analysis/fault_analysis_record.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include <algorithm>

namespace hozon {
namespace netaos {
namespace phm_server {

#define PHMS_DEBUG_ANALYSIS if(debug_switch_ == "on")PHMS_DEBUG


FaultAnalysis *FaultAnalysis::instancePtr_ = nullptr;
std::mutex FaultAnalysis::mtx_;
std::mutex FaultAnalysis::update_analysisfile_mtx_;

FaultAnalysis *FaultAnalysis::getInstance()
{
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new FaultAnalysis();
        }
    }
    return instancePtr_;
}

FaultAnalysis::FaultAnalysis()
: time_mgr_(new TimerManager())
, init_timer_fd_(-1)
{
}

FaultAnalysis::~FaultAnalysis()
{

}

void
FaultAnalysis::Init()
{
    PHMS_INFO << "FaultAnalysis::Init enter!";
    FaultTaskHandler::getInstance()->RegisterAnalysisCallback(std::bind(&FaultAnalysis::AnalysisFaultCallback, this, std::placeholders::_1));
    FaultAnalysisRecord::getInstance()->Init();

    analy_list_.clear();
    analy_status_list_.clear();
    analy_nonstandard_list_.clear();
    system_check_list_.clear();
    current_fault_list_.clear();
    debug_switch_ = PHMServerConfig::getInstance()->GetPhmConfigInfo().DebugSwitch;
    const unsigned int updateAnalysisFileTime = PHMServerConfig::getInstance()->GetPhmConfigInfo().AnalysisTime;

    // init timer manager
    if (time_mgr_ != nullptr) {
        time_mgr_->Init();
        time_mgr_->StartFdTimer(init_timer_fd_, updateAnalysisFileTime,
            std::bind(&FaultAnalysis::InitTimeoutCallback, this, std::placeholders::_1), NULL, true);
    }
}

void
FaultAnalysis::DeInit()
{
    PHMS_INFO << "FaultAnalysis::DeInit enter!";
    FaultAnalysisRecord::getInstance()->StartRecordAnalyDataToFile(GetFaultAnalyData());

    if (time_mgr_ != nullptr) {
        PHMS_INFO << "FaultAnalysis::DeInit fd:" << init_timer_fd_;
        time_mgr_->StopFdTimer(init_timer_fd_);
        time_mgr_->DeInit();
    }

    FaultAnalysisRecord::getInstance()->DeInit();
    if (instancePtr_ != nullptr) {
        delete instancePtr_;
        instancePtr_ = nullptr;
    }
}

void
FaultAnalysis::InitTimeoutCallback(void* data)
{
    PHMS_INFO << "FaultAnalysis::InitTimeoutCallback enter!";
    std::unique_lock<std::mutex> lck(update_analysisfile_mtx_);
    FaultAnalysisRecord::getInstance()->StartRecordAnalyDataToFile(GetFaultAnalyData());
}

void
FaultAnalysis::AnalysisFaultCallback(Fault_t fault)
{
    PHMS_INFO << "FaultAnalysis::AnalysisFaultCallback fault " << fault.faultId*100 + fault.faultObj
              << " analysis: " << fault.faultAction.analysis;
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    if ("off" == configInfo.AnalysisSwitch) {
        return;
    }

    if (!fault.faultAction.analysis) {
        return;
    }

    NotifyAnalysisFaultData(fault);
}

void
FaultAnalysis::NotifyAnalysisFaultData(const Fault_t& faultData)
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::NotifyAnalysisFaultData key " << faultData.faultId*100 + faultData.faultObj;
    for (uint32_t i = 0; i < analy_list_.size(); ++i) {
        if (faultData.faultProcess == analy_list_[i].faultProcess) {
            PHMS_DEBUG_ANALYSIS << "FaultAnalysis::NotifyAnalysisFaultData faultProcess " << faultData.faultProcess << " update!";
            UpdateAnalyList(faultData, analy_list_[i]);
            return;
        }
    }

    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::NotifyAnalysisFaultData faultProcess " << faultData.faultProcess << " is new!";
    SaveFaultToAnalyList(faultData);


    for (uint32_t i = 0; i < analy_status_list_.size(); i++) {
        if (faultData.faultDomain == analy_status_list_[i].faultDomain) {
            PHMS_DEBUG_ANALYSIS << "FaultAnalysis::NotifyAnalysisFaultData faultDomain :" << analy_status_list_[i].faultDomain << " update!";
            UpdateAnalyStatusList(faultData, analy_status_list_[i]);
            return;
        }
    }

    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::NotifyAnalysisFaultData faultDomain" << faultData.faultDomain << " is new!";
    SaveAnalyStatusList(faultData);
    return;
}

void
FaultAnalysis::UpdateAnalyFile()
{
    std::unique_lock<std::mutex> lck(update_analysisfile_mtx_);
    FaultAnalysisRecord::getInstance()->StartRecordAnalyDataToFile(GetFaultAnalyData());
}

void
FaultAnalysis::UpdateAnalyList(const Fault_t& recvFaultData, AnalysisFault& savedFaultData)
{
    if (0x00 == recvFaultData.faultStatus) {
        savedFaultData.recoverCount++;
        return;
    }

    savedFaultData.matureCount += 1;
    if (recvFaultData.faultLevel == 0x03) {
        savedFaultData.fatalCount += 1;
    }
    else if (recvFaultData.faultLevel == 0x02) {
        savedFaultData.criticalCount += 1;
    }
    else if (recvFaultData.faultLevel == 0x01) {
        savedFaultData.normalCount += 1;
    }
    else {
        savedFaultData.infoCount += 1;
    }

    uint64_t currTime = recvFaultData.faultOccurTime / 1000000;
    uint64_t diffTime = 0;
    if (currTime < savedFaultData.faultOccurTime) {
        savedFaultData.faultOccurTime = currTime;
        diffTime = savedFaultData.faultOccurTime - currTime;
    }
    else {
        diffTime = currTime - savedFaultData.faultOccurTime;
    }

    savedFaultData.avgCyc = diffTime / (savedFaultData.matureCount - 1);
    if (savedFaultData.minGap != 0) {
        savedFaultData.minGap = (diffTime < savedFaultData.minGap) ? diffTime : savedFaultData.minGap;
        savedFaultData.maxGap = (diffTime > savedFaultData.maxGap) ? diffTime : savedFaultData.maxGap;
    }
    else {
        savedFaultData.minGap = savedFaultData.avgCyc;
        savedFaultData.maxGap = savedFaultData.avgCyc;
    }

    return;
}

void
FaultAnalysis::SaveFaultToAnalyList(const Fault_t& faultData)
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::SaveFaultToAnalyList new fault key " << faultData.faultId*100 + faultData.faultObj;
    if (0x00 == faultData.faultStatus) {
        return;
    }

    AnalysisFault analyFaultData1;
    analyFaultData1.faultProcess = faultData.faultProcess;
    analyFaultData1.faultDscribe = faultData.faultDscribe;
    analyFaultData1.faultDomain = faultData.faultDomain;
    analyFaultData1.key = faultData.faultId*100 + faultData.faultObj;
    analyFaultData1.faultOccurTime = faultData.faultOccurTime / 1000000;
    analyFaultData1.matureCount = 1;
    analyFaultData1.recoverCount = 0;
    analyFaultData1.avgCyc = 0;
    analyFaultData1.minGap = 0;
    analyFaultData1.maxGap = 0;
    analyFaultData1.fatalCount = 0;
    analyFaultData1.criticalCount = 0;
    analyFaultData1.normalCount = 0;
    analyFaultData1.infoCount = 0;
    if (faultData.faultLevel == 0x03) {
        analyFaultData1.fatalCount = 1;
    }
    else if (faultData.faultLevel == 0x02) {
        analyFaultData1.criticalCount = 1;
    }
    else if (faultData.faultLevel == 0x01) {
        analyFaultData1.normalCount = 1;
    }
    else {
        analyFaultData1.infoCount = 1;
    }

    analy_list_.push_back(analyFaultData1);
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::SaveFaultToAnalyList new fault analy_list_ size " << analy_list_.size();
    return;
}

void
FaultAnalysis::UpdateAnalyStatusList(const Fault_t& faultData, AnalysisFaultStatus& analysisFaultStatus)
{
    if (faultData.faultStatus != analysisFaultStatus.faultStatus) {
        analysisFaultStatus.faultStatus = faultData.faultStatus;
        analysisFaultStatus.count += 1;
    }
    else {
        // status equel is invalid operation and update analy_nonstandard_list_
        PHMS_INFO << "FaultAnalysis::UpdateAnalyStatusList";
        UpdateAnalyNonstandardList(faultData.faultDomain, analysisFaultStatus);
    }

    return;
}

void
FaultAnalysis::SaveAnalyStatusList(const Fault_t& faultData)
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::SaveAnalyStatusList new analyResult! index 1 key " << faultData.faultId*100 + faultData.faultObj;
    AnalysisFaultStatus analyStatusData;
    analyStatusData.key = faultData.faultId*100 + faultData.faultObj;
    analyStatusData.faultProcess = faultData.faultProcess;
    analyStatusData.faultDomain = faultData.faultDomain;
    analyStatusData.faultStatus = faultData.faultStatus;
    analyStatusData.count = 1;

    analy_status_list_.push_back(analyStatusData);
    return;
}

void
FaultAnalysis::UpdateAnalyNonstandardList(const std::string faultDomain, AnalysisFaultStatus& analysisFaultStatus)
{
    for (uint32_t j = 0; j < analy_nonstandard_list_.size(); j++) {
        if (faultDomain == analy_nonstandard_list_[j].faultDomain) {
            PHMS_DEBUG_ANALYSIS << "FaultAnalysis::UpdateAnalyNonstandardList faultDomain :" << analysisFaultStatus.faultDomain << " update!";
            analy_nonstandard_list_[j].count++;
            return;
        }
    }

    AnalisysNonstandard analysNonstandardData;
    analysNonstandardData.faultProcess = analysisFaultStatus.faultProcess;
    analysNonstandardData.faultDomain = analysisFaultStatus.faultDomain;
    analysNonstandardData.key = analysisFaultStatus.key;
    analysNonstandardData.count = analysisFaultStatus.count + 1;
    analy_nonstandard_list_.push_back(analysNonstandardData);
    return;
}

std::string
FaultAnalysis::ReadAnalyCountData()
{
    PHMS_INFO << "FaultAnalysis::ReadAnalyCountData enter!";
    std::string str = "Fault Count Analysis\n";
    str += "PROCESS                      MATURE   RECOVER     FATAL  CRITICAL    NORMAL      INFO      AvgCyc/ms      MinGap/ms      MaxGap/ms\n";
    char buff[256] = {0};
    int index = 0;
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyCountData analy_list_ size " << analy_list_.size();
    if (analy_list_.empty()) {
        PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyCountData analy_list_ is empty!";
        return str;
    }

    AnalisysOverCount analysOverCountData;
    analy_over_count_list_.clear();
    for (auto & item: analy_list_) {
        memset(buff, 0, sizeof(buff));
        index = snprintf(buff, 256 - 1, "%-25s%10d%10d%10d%10d%10d%10d%15lu%15lu%15lu\n", item.faultProcess.c_str(), item.matureCount, item.recoverCount,
            item.fatalCount, item.criticalCount, item.normalCount, item.infoCount, item.avgCyc, item.minGap, item.maxGap);
        // PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyCountData buff: " << buff;
        if (index > 0) {
            str = str + std::string(buff);
        }

        if (item.matureCount > 500) {
            analysOverCountData.faultDomain = item.faultDomain;
            analysOverCountData.key = item.key;
            analysOverCountData.faultDscribe = item.faultDscribe;
            analysOverCountData.count = item.matureCount;
            analysOverCountData.faultProcess = item.faultProcess;
            analy_over_count_list_.push_back(analysOverCountData);
        }
    }

    return str;
}

std::string
FaultAnalysis::ReadAnalysNonstandardData()
{
    PHMS_INFO << "FaultAnalysis::ReadAnalysNonstandardData enter!";
    std::string str = "Sync Fault Nonstandard Report Analysis\n";
    str += "Domain                     Fault     Count\n";
    char buff[64] = {0};
    int index = 0;
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalysNonstandardData analy_nonstandard_list_ size " << analy_nonstandard_list_.size();
    if (analy_nonstandard_list_.empty()) {
        return str;
    }

    for (auto & item: analy_nonstandard_list_) {
        memset(buff, 0, sizeof(buff));
        index = snprintf(buff, 64 - 1, "%-25s%7d%10d\n", item.faultDomain.c_str(), item.key, item.count);
        // PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalysNonstandardData buff: " << buff;
        if (index > 0) {
            str = str + std::string(buff);
        }
    }

    return str;
}

std::string
FaultAnalysis::ReadAnalyOverCountData()
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyOverCountData ";
    std::string str = "Over Count Fault\n";
    str += "Domain                   Dscribe                                                                           Fault     Count\n";
    char buff[256] = {0};
    int index = 0;
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyOverCountData analy_over_count_list_ size " << analy_over_count_list_.size();
    if (analy_over_count_list_.empty()) {
        return str;
    }

    for (auto & item: analy_over_count_list_) {
        memset(buff, 0, sizeof(buff));
        index = snprintf(buff, 256 - 1, "%-25s%-80s%7d%10d\n", item.faultDomain.c_str(), item.faultDscribe.c_str(), item.key, item.count);
        // PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadAnalyOverCountData buff: " << buff;
        if (index > 0) {
            str = str + std::string(buff);
        }
    }

    return str;
}

std::string
FaultAnalysis::ReadStartupFaultData()
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadStartupFaultData ";
    std::string str = "Startup Phase Inhibit Window Faults\n";
    str += "  Fault     Count   last status\n";
    char buff[64] = {0};
    int index = 0;
    FaultDispatcher::getInstance()->QuerySystemCheckFault(system_check_list_);
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadStartupFaultData system_check_list_ size " << system_check_list_.size();
    if (system_check_list_.empty()) {
        return str;
    }

    for (auto & item: system_check_list_) {
        memset(buff, 0, sizeof(buff));
        index = snprintf(buff, 64 - 1, "%7d%10d%8d\n", (item.faultId*100 + item.faultObj), item.faultOccurCount, item.faultStatus);
        // PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadStartupFaultData buff: " << buff;
        if (index > 0) {
            str = str + std::string(buff);
        }
    }

    return str;
}

std::string
FaultAnalysis::ReadPresentFaultData()
{
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadPresentFaultData ";
    std::string str = "Present Occur Fault\n";
    char buff[32] = {0};
    int index = 0;
    int i = 0;

    PHMServerConfig::getInstance()->QueryAllOccuredFault(current_fault_list_);
    PHMS_DEBUG_ANALYSIS << "FaultAnalysis::ReadPresentFaultData current_fault_list_ size " << current_fault_list_.size();
    if (current_fault_list_.empty()) {
        return str;
    }

    sort(current_fault_list_.begin(), current_fault_list_.end());
    for (auto & item: current_fault_list_) {
        ++i;
        memset(buff, 0, sizeof(buff));
        index = snprintf(buff, 32 - 1, "%7d ", item);
        // PHM_LOG_DEBUG_ANALYSIS << "HzFMAnalysis::ReadPresentFaultData buff: " << buff;
        if (index > 0) {
            str = str + std::string(buff);
        }

        if (0 == (i % 10)) {
            str = str + std::string("\n");
        }
    }

    return str;
}

std::string
FaultAnalysis::GetFaultAnalyData()
{
    // fault analy data
    std::string faultAnalyData = "";
    faultAnalyData += ReadAnalyCountData() + "\n";
    faultAnalyData += ReadAnalysNonstandardData() + "\n";
    faultAnalyData += ReadAnalyOverCountData() + "\n";
    faultAnalyData += ReadStartupFaultData() + "\n";
    faultAnalyData += ReadPresentFaultData() + "\n";
    return faultAnalyData;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
