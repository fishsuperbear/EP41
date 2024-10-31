/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: phm fault dispatcher
 */
#include <errno.h>
#include <string.h>
#include <cstdint>
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_utils.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/fault_manager/analysis/fault_analysis.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include "phm_server/include/fault_lock/phm_fault_lock.h"

namespace hozon {
namespace netaos {
namespace phm_server {

FaultDispatcher* FaultDispatcher::instance_ = nullptr;
std::mutex FaultDispatcher::mtx_;
const std::string fault_event_topic = "fault_eventTopic";
const std::string fault_inhibit_topic = "faultInhibitEventTopic";

FaultDispatcher*
FaultDispatcher::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new FaultDispatcher();
        }
    }

    return instance_;
}

FaultDispatcher::FaultDispatcher()
: proxy_(nullptr)
, skeleton_(nullptr)
, time_mgr_(new TimerManager())
, system_check_timer_fd_(-1)
, system_check_completed_(false)
, inhibitType_(0)
{
}

void
FaultDispatcher::Init()
{
    PHMS_INFO << "FaultDispatcher::Init";
    // init timer manager
    if (time_mgr_ != nullptr) {
        time_mgr_->Init();
    }

    // init proxy
    std::shared_ptr<fault_eventPubSubType> proxyPubsubtype = std::make_shared<fault_eventPubSubType>();
    proxy_ = std::make_shared<Proxy>(proxyPubsubtype);
    proxy_->Init(0, fault_event_topic);
    proxy_->Listen(std::bind(&FaultDispatcher::ReceiveFault, this));

    // init skeleton
    std::shared_ptr<fault_eventPubSubType> skeletonPubsubtype = std::make_shared<fault_eventPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(skeletonPubsubtype);
    skeleton_->Init(0, fault_event_topic);

    std::shared_ptr<faultInhibitEventPubSubType> skeletonInhibitTypePubsubtype = std::make_shared<faultInhibitEventPubSubType>();
    inhibit_skeleton_ = std::make_shared<Skeleton>(skeletonInhibitTypePubsubtype);
    inhibit_skeleton_->Init(0, fault_inhibit_topic);

    // system check
    SystemCheck();
}

void
FaultDispatcher::DeInit()
{
    PHMS_INFO << "FaultDispatcher::DeInit";
    inhibitType_ = 0;
    system_check_timer_fd_ = -1;
    system_check_fault_list_.clear();

    // deinit skeleton
    if (skeleton_ != nullptr) {
        skeleton_->Deinit();
        skeleton_ = nullptr;
    }

    // deinit proxy
    if (proxy_ != nullptr) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }

    // timer manager deinit
    if (time_mgr_ != nullptr) {
        time_mgr_->DeInit();
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
FaultDispatcher::SystemCheck()
{
    PHMS_INFO << "FaultDispatcher::SystemCheck";
    const PhmConfigInfo& configInfo = PHMServerConfig::getInstance()->GetPhmConfigInfo();
    if (configInfo.SystemCheckTime > 0) {
        if (time_mgr_ != nullptr) {
            time_mgr_->StartFdTimer(system_check_timer_fd_, configInfo.SystemCheckTime,
                std::bind(&FaultDispatcher::SystemCheckCompletedCallback, this, std::placeholders::_1), NULL, false);
        }
    }
    else {
        system_check_completed_ = true;
    }
}

void
FaultDispatcher::SystemCheckCompletedCallback(void* data)
{
    PHMS_INFO << "FaultDispatcher::SystemCheckCompletedCallback";
    system_check_completed_ = true;

    for (auto& item : system_check_fault_list_) {
        Fault_t fault;
        fault.faultId = item.second.faultId;
        fault.faultObj = item.second.faultObj;
        fault.faultStatus = item.second.faultStatus;
        fault.faultOccurTime = item.second.faultOccurTime;
        fault.faultDomain = item.second.faultDomain;

        ReportFault(fault);
    }
}

void
FaultDispatcher::ReceiveFault()
{
    if (nullptr == proxy_) {
        PHMS_ERROR << "FaultDispatcher::ReceiveFault proxy_ is nullptr.";
        return;
    }

    if (!proxy_->IsMatched()) {
        PHMS_WARN << "FaultDispatcher::ReceiveFault proxy_ not matched.";
        return;
    }

    std::shared_ptr<fault_event> data = std::make_shared<fault_event>();
    proxy_->Take(data);

    uint32_t fault_key = data->fault_id()*100 + data->fault_obj();
    if (fault_status_maps_.count(fault_key) > 0 && fault_status_maps_[fault_key] == data->fault_status()) {
        PHMS_DEBUG << "FaultDispatcher::ReceiveFault same fault status, ignore this fault: " << fault_key;
        return;
    }
    else {
        fault_status_maps_[fault_key] = data->fault_status();
    }

    PHMS_INFO << "FaultDispatcher::ReceiveFault domain: " << data->domain() << " occur_time: " << data->occur_time() \
                << " fault_id: " << data->fault_id() << " fault_obj: " << data->fault_obj() \
                << " fault_status: " << data->fault_status();
    // fault report
    Fault_t fault;
    fault.faultId = data->fault_id();
    fault.faultObj = data->fault_obj();
    fault.faultStatus = data->fault_status();
    fault.faultOccurTime = data->occur_time();
    fault.faultDomain = data->domain();
    ReportFault(fault);
    // fault lock report
    // FaultLockReportInfo faultLock;
    // faultLock.faultId = data->fault_id();
    // faultLock.faultObj = data->fault_obj();
    // faultLock.faultStatus = data->fault_status();
    // faultLock.isInhibitWindowFault = !system_check_completed_;
    // FaultLock::getInstance()->ReportFault(faultLock);
}

void
FaultDispatcher::SendFault(const SendFaultPack& fault)
{
    if (nullptr == skeleton_) {
        PHMS_ERROR << "FaultDispatcher::SendFault skeleton_ is nullptr.";
        return;
    }

    if (!skeleton_->IsMatched()) {
        PHMS_WARN << "FaultDispatcher::SendFault skeleton_ not matched.";
        return;
    }

    uint32_t fault_key = fault.faultId * 100 + fault.faultObj;
    if (fault_status_maps_.count(fault_key) > 0 && fault_status_maps_[fault_key] == fault.faultStatus) {
        PHMS_DEBUG << "FaultDispatcher::SendFault same fault status, ignore this fault: " << fault_key;
        return;
    }
    else {
        fault_status_maps_[fault_key] = fault.faultStatus;
    }

    std::shared_ptr<fault_event> data = std::make_shared<fault_event>();
    data->domain(fault.faultDomain);
    data->occur_time(fault.faultOccurTime);
    data->fault_id(fault.faultId);
    data->fault_obj(fault.faultObj);
    data->fault_status(fault.faultStatus);

    PHMS_INFO << "FaultDispatcher::SendFault -> domain: " << data->domain() << " occur_time: " << data->occur_time() \
            << " fault_id: " << data->fault_id() << " fault_obj: " << data->fault_obj() \
            << " fault_status: " << data->fault_status();
    int32_t res = skeleton_->Write(data);
    if (res != 0) {
        PHMS_INFO << "FaultDispatcher::SendFault fault key " << fault_key << " send failed!" << "|res:" << res;
    }
}

void
FaultDispatcher::ReportFault(const Fault_t& fault)
{
    std::lock_guard<std::mutex> lck(mtx_);
    Fault_t cFaultInfo;
    uint32_t faultKey = fault.faultId * 100 + fault.faultObj;
    PHMS_DEBUG << "FaultDispatcher::ReportFault domain: "<< fault.faultDomain << " fault key: " << faultKey;
    if (!PHMServerConfig::getInstance()->GetFaultInfoByFault(faultKey, cFaultInfo)) {
        PHMS_ERROR << "FaultDispatcher::ReceiveFault error fault: " << faultKey;
        return;
    }
    cFaultInfo.faultStatus = fault.faultStatus;
    cFaultInfo.faultOccurTime = fault.faultOccurTime;
    cFaultInfo.faultDomain = fault.faultDomain;

    if (!system_check_completed_) {
        PHMS_DEBUG << "FaultDispatcher::ReceiveFault system is still checking.";
        AddSystemCheckFault(cFaultInfo);
        return;
    }

    if (0 != inhibitType_) {
        PHMS_WARN << "FaultDispatcher::ReceiveFault inhibit type:" << inhibitType_;
        return;
    }

    // update fault status
    PHMServerConfig::getInstance()->UpdateFaultStatus(faultKey, cFaultInfo);

    // send fault to task
    FaultTask task;
    if (cFaultInfo.faultAction.record) {
        task.type_list.emplace_back(kRecord);
    }

    if (cFaultInfo.faultAction.analysis) {
        task.type_list.emplace_back(kAnalysis);
    }

    if (cFaultInfo.faultAction.strategy.notifyMcu
        || cFaultInfo.faultAction.strategy.notifyApp
        || cFaultInfo.faultAction.strategy.restartproc
        || cFaultInfo.faultAction.strategy.dtcMapping) {
        task.type_list.emplace_back(kStrategy);
    }

    task.fault = cFaultInfo;
    FaultTaskHandler::getInstance()->AddFault(task);
}

void
FaultDispatcher::AddSystemCheckFault(const Fault_t& fault)
{
    uint32_t key = fault.faultId + fault.faultObj;
    auto itr = system_check_fault_list_.find(key);
    if (itr != system_check_fault_list_.end()) {
        itr->second.faultDomain = fault.faultDomain;
        itr->second.faultOccurTime = fault.faultOccurTime;
        itr->second.faultStatus = fault.faultStatus;
        if (itr->second.faultStatus) {
            itr->second.faultOccurCount++;
        }
        else {
            itr->second.faultRecoverCount++;
        }

        return;
    }

    SystemCheckFaultInfo faultInfo;
    faultInfo.faultDomain = fault.faultDomain;
    faultInfo.faultOccurTime = fault.faultOccurTime;
    faultInfo.faultId = fault.faultId;
    faultInfo.faultObj = fault.faultObj;
    faultInfo.faultStatus = fault.faultStatus;
    if (faultInfo.faultStatus) {
        faultInfo.faultOccurCount = 1;
        faultInfo.faultRecoverCount = 0;
    }
    else {
        faultInfo.faultRecoverCount = 1;
        faultInfo.faultOccurCount = 0;
    }

    system_check_fault_list_.insert(std::make_pair(key, faultInfo));
}

void
FaultDispatcher::QuerySystemCheckFault(std::vector<SystemCheckFaultInfo>& faultList)
{
    PHMS_DEBUG << "FaultDispatcher::QuerySystemCheckFault";
    faultList.clear();
    if (!system_check_completed_) {
        PHMS_DEBUG << "FaultDispatcher::QuerySystemCheckFault system is still checking.";
        return;
    }

    for (auto& item : system_check_fault_list_) {
        faultList.push_back(item.second);
    }
}

void
FaultDispatcher::SendInhibitType(const uint32_t type)
{
    PHMS_DEBUG << "FaultDispatcher::SendInhibitType type:" << type;
    if (nullptr == inhibit_skeleton_) {
        PHMS_ERROR << "FaultDispatcher::SendInhibitType inhibit_skeleton_ is nullptr.";
        return;
    }

    if (!inhibit_skeleton_->IsMatched()) {
        PHMS_WARN << "FaultDispatcher::SendInhibitType inhibit_skeleton_ not matched.";
        return;
    }

    std::shared_ptr<faultInhibitEvent> data = std::make_shared<faultInhibitEvent>();
    data->type(type);
    int32_t res = inhibit_skeleton_->Write(data);
    if (res != 0) {
        PHMS_INFO << "FaultDispatcher::SendInhibitType type:" << data->type() << " send failed!" << "|res:" << res;
    }

    inhibitType_ = type;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
