/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault event receive
*/
#include "phm/fault_manager/include/fault_inhibit_judge.h"
#include "phm/fault_manager/include/fault_dispatcher.h"
#include "phm/common/include/phm_config.h"
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/timer_manager.h"
#include <unordered_set>

namespace hozon {
namespace netaos {
namespace phm {

const std::string FAULT_EVENT_TOPIC = "fault_eventTopic";
const std::string FAULT_INHIBIT_EVENT = "faultInhibitEventTopic";
const std::unordered_set<uint32_t> IGNORE_FAULT_SET = { 492001, 492002, 492003 };

FaultDispatcher::FaultDispatcher()
: system_check_timer_fd_(-1)
, system_check_completed_(false)
, stopFlag_(false)
{
}

FaultDispatcher::~FaultDispatcher()
{
}

int32_t
FaultDispatcher::Init(std::function<void(ReceiveFault_t)> fault_receive_callback)
{
    PHM_INFO << "FaultDispatcher::Init start..";
    fault_receive_callback_ = fault_receive_callback;
    pubsubtype_ = std::make_shared<fault_eventPubSubType>();

    int32_t res1 = InitSend();
    int32_t res2 = InitRecv();
    int32_t res3 = InitInhibitTypeRecv();

    // start inhibit timer
    SystemCheck();

    PHM_INFO << "FaultDispatcher::Init end.. res1[" << res1 << "] res2[" << res2 << "] res3[" << res3 << "].";
    if (0 == res1 && 0 == res2 && 0 == res3) {
        std::function<void(void)> functions[THREAD_NUMS] = {
            std::bind(&FaultDispatcher::FaultSendThread, this)
        };

        for (int i = 0; i < THREAD_NUMS; i++) {
            dispatcher_threads_[i] = std::thread(functions[i]);
        }
        return 0;
    }

    return -1;
}

void
FaultDispatcher::Deinit()
{
    stopFlag_ = true;
    NotifyAll();

    for (int i = 0; i < THREAD_NUMS; i++) {
        if(dispatcher_threads_[i].joinable()) {
            dispatcher_threads_[i].join();
        }
    }

    if (inhibit_type_proxy_ != nullptr) {
        inhibit_type_proxy_->Deinit();
        inhibit_type_proxy_ = nullptr;
    }

    if (proxy_ != nullptr) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }

    if (skeleton_) {
        skeleton_->Deinit();
        skeleton_ = nullptr;
    }
}

int32_t
FaultDispatcher::InitSend()
{
    PHM_INFO << "FaultDispatcher::InitSend";
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    return skeleton_->Init(0, FAULT_EVENT_TOPIC);
}

int32_t
FaultDispatcher::InitRecv()
{
    PHM_INFO << "FaultDispatcher::InitRecv";
    proxy_ = std::make_shared<Proxy>(pubsubtype_);
    int32_t res = proxy_->Init(0, FAULT_EVENT_TOPIC);
    if (0 == res) {
        event_data_ = std::make_shared<fault_event>();
        proxy_->Listen(std::bind(&FaultDispatcher::Recv, this));
    }

    return res;
}

void
FaultDispatcher::NotifyAll()
{
    for (int i = 0; i < THREAD_NUMS; i++) {
        cvs_[i].notify_one();
    }
}

int32_t
FaultDispatcher::InitInhibitTypeRecv()
{
    PHM_INFO << "FaultDispatcher::InitInhibitTypeRecv";
    fault_inhibit_data_ = std::make_shared<faultInhibitEvent>();
    fault_inhibit_pubsubtype_ = std::make_shared<faultInhibitEventPubSubType>();
    inhibit_type_proxy_ = std::make_shared<Proxy>(fault_inhibit_pubsubtype_);
    if (inhibit_type_proxy_ == nullptr) {
        PHM_ERROR << "FaultDispatcher::InitInhibitTypeRecv inhibit_type_proxy_ is null";
        return -1;
    }

    int32_t res = inhibit_type_proxy_->Init(0, FAULT_INHIBIT_EVENT);
    if (0 == res) {
        inhibit_type_proxy_->Listen(std::bind(&FaultDispatcher::InhibitTypeCb, this));
    }

    return res;
}

void
FaultDispatcher::InhibitTypeCb()
{
    if (inhibit_type_proxy_ != nullptr && inhibit_type_proxy_->IsMatched()) {
        inhibit_type_proxy_->Take(fault_inhibit_data_);
        PHM_DEBUG << "FaultDispatcher::InhibitTypeCb type:" << fault_inhibit_data_->type();
        FaultInhibitJudge* pcFaultInhibitJudge = FaultInhibitJudge::getInstance();
        pcFaultInhibitJudge->SetInhibitType(fault_inhibit_data_->type());
    }
}

void
FaultDispatcher::PushFaultQueue(SendFaultPack_t& sendFault)
{
    PHM_TRACE << "FaultDispatcher::PushFaultQueue enter!";
    if(faultq_.size() > QUEUE_DEPTH) {
        PHM_WARN << "The faultq_ is full. Current fault event discard! faultid: " \
            << sendFault.faultId << " faultobj: " << sendFault.faultObj;
        return;
    }

    {
        std::unique_lock<std::mutex> lck(mutexs_[0]);
        faultq_.push(sendFault);
    }
    cvs_[0].notify_one();
}

void
FaultDispatcher::FaultSendThread()
{
    while(!stopFlag_) {
        std::unique_lock<std::mutex> lck(mutexs_[0]);
        cvs_[0].wait(lck);
        while(!faultq_.empty()) {
            SendFaultPack_t sendData = faultq_.front();
            faultq_.pop();
            // lck.unlock();

            LocalRecv(sendData);
        }
    }
}

void
FaultDispatcher::Send(SendFaultPack_t& sendFault)
{
    std::shared_ptr<fault_event> data = std::make_shared<fault_event>();
    data->domain(sendFault.faultDomain);
    data->occur_time(sendFault.faultOccurTime);
    data->fault_id(sendFault.faultId);
    data->fault_obj(sendFault.faultObj);
    data->fault_status(sendFault.faultStatus);

    if ((skeleton_ == nullptr) || (!skeleton_->IsMatched())) {
        PHM_WARN << "FaultDispatcher::Send end.. But skeleton is null or skeleton not matched!";
        return;
    }

    PHM_INFO << "FaultDispatcher::Send -> domain: " << data->domain() << ", occur_time: " << data->occur_time()
        << ", fault_id: " << data->fault_id() << ", fault_obj: " << (int)data->fault_obj()
        << ", fault_status: " << (int)data->fault_status();
    int32_t res = skeleton_->Write(data);
    if (res != 0) {
        PHM_WARN << "FaultDispatcher::Send id: " << data->fault_id() << ", obj: " << (int)data->fault_obj() << "|res:" << res;
    }
}

void
FaultDispatcher::Recv(void)
{
    if (proxy_ != nullptr && proxy_->IsMatched()) {
        proxy_->Take(event_data_);
        uint32_t key = event_data_->fault_id() * 100 + event_data_->fault_obj();
        bool isIgnoreFaultKey = (IGNORE_FAULT_SET.count(key) > 0 ? true : false);

        if ((!system_check_completed_) && (!isIgnoreFaultKey)) {
            PHM_DEBUG << "FaultDispatcher::Recv system is still checking.";
            return;
        }

        if (local_receive_flag_.count(key) > 0 && (true == local_receive_flag_[key])) {
            local_receive_flag_[key] = false;
            return;
        }

        PhmFaultInfo faultInfo;
        if (!PHMConfig::getInstance()->GetFaultInfoByFault(key, faultInfo)) {
            PHM_ERROR << "FaultReceiver::Recv error fault: " << key;
            return;
        }

        ReceiveFault_t recv_fault;
        recv_fault.faultId = event_data_->fault_id();
        recv_fault.faultObj = event_data_->fault_obj();
        recv_fault.faultStatus = event_data_->fault_status();
        recv_fault.faultDomain = event_data_->domain();
        recv_fault.faultOccurTime = event_data_->occur_time();
        recv_fault.faultDes = faultInfo.faultDescribe;
        recv_fault.faultCombinationId = faultInfo.faultClusterId;

        std::vector<FaultReceiveMap> maps;
        FaultReceiveTable::getInstance()->GetAllMap(recv_fault, maps);

        for (uint32_t i = 0; i< maps.size(); ++i) {
            if (maps[i].cfg->IsInhibitFault(key)) {
                PHM_INFO << "FaultReceiver::Recv inhibit fault: " << key;
                continue;
            }

            PHMConfig::getInstance()->UpdateFaultStatus(key, event_data_->fault_status());
            PHMConfig::getInstance()->GetRegistCluster(recv_fault.faultCluster, maps[i].cfg);

            PHM_INFO << "FaultDispatcher::Recv need callback faultId: " << recv_fault.faultId
                        << " faultObj: " << (int)recv_fault.faultObj
                        << " faultStatus: " << (int)recv_fault.faultStatus
                        << " faultOccurTime: " << recv_fault.faultOccurTime
                        << " faultDomain: " << recv_fault.faultDomain
                        << " faultDes: " << recv_fault.faultDes;

            if (nullptr != maps[i].recv_callback) {
                maps[i].recv_callback(recv_fault);
            }
        }
    }
}

void
FaultDispatcher::LocalRecv(SendFaultPack_t& sendFault)
{
    PHM_TRACE << "FaultDispatcher::LocalRecv ..";
    if (!system_check_completed_) {
        PHM_DEBUG << "FaultDispatcher::LocalRecv system is still checking.";
        return;
    }

    local_receive_flag_[sendFault.faultId * 100 + sendFault.faultObj] = true;

    uint32_t key = sendFault.faultId * 100 + sendFault.faultObj;
    PhmFaultInfo faultInfo;
    if (!PHMConfig::getInstance()->GetFaultInfoByFault(key, faultInfo)) {
        PHM_ERROR << "FaultReceiver::LocalRecv error fault: " << key;
        return;
    }

    ReceiveFault_t recv_fault;
    recv_fault.faultDomain = sendFault.faultDomain;
    recv_fault.faultId = sendFault.faultId;
    recv_fault.faultObj = sendFault.faultObj;
    recv_fault.faultOccurTime = sendFault.faultOccurTime;
    recv_fault.faultStatus = sendFault.faultStatus;
    recv_fault.faultDes = faultInfo.faultDescribe;
    recv_fault.faultCombinationId = faultInfo.faultClusterId;

    std::vector<FaultReceiveMap> maps;
    FaultReceiveTable::getInstance()->GetAllMap(recv_fault, maps);

    for (uint32_t i = 0; i< maps.size(); ++i) {
        if (maps[i].cfg->IsInhibitFault(key)) {
            PHM_INFO << "FaultReceiver::LocalRecv inhibit fault: " << key;
            continue;
        }

        PHMConfig::getInstance()->UpdateFaultStatus(key, event_data_->fault_status());
        PHMConfig::getInstance()->GetRegistCluster(recv_fault.faultCluster, maps[i].cfg);

        PHM_INFO << "FaultDispatcher::LocalRecv need callback faultId: " << recv_fault.faultId
                    << " faultObj: " << (int)recv_fault.faultObj
                    << " faultStatus: " << (int)recv_fault.faultStatus
                    << " faultOccurTime: " << recv_fault.faultOccurTime
                    << " faultDomain: " << recv_fault.faultDomain
                    << " faultDes: " << recv_fault.faultDes;

        if (nullptr != maps[i].recv_callback) {
            maps[i].recv_callback(recv_fault);
        }
    }

    return;
}

void
FaultDispatcher::SystemCheck()
{
    PHM_INFO << "FaultDispatcher::SystemCheck";
    const PhmConfigInfo& configInfo = PHMConfig::getInstance()->GetPhmConfigInfo();
    if (configInfo.SystemCheckTime > 0) {
        TimerManager::Instance()->StartFdTimer(system_check_timer_fd_, configInfo.SystemCheckTime,
            std::bind(&FaultDispatcher::SystemCheckCompletedCallback, this, std::placeholders::_1), NULL, false);
    }
    else {
        system_check_completed_ = true;
    }
}

void
FaultDispatcher::SystemCheckCompletedCallback(void* data)
{
    PHM_INFO << "FaultDispatcher::SystemCheckCompletedCallback";
    system_check_completed_ = true;
    if (system_check_timer_fd_ != -1) {
        TimerManager::Instance()->StopFdTimer(system_check_timer_fd_);
    }

    system_check_timer_fd_ = -1;
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
