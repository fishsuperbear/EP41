/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server transport service
*/

#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerTransPortService* DiagServerTransPortService::instance_ = nullptr;
std::mutex DiagServerTransPortService::mtx_;

using namespace hozon::netaos::diag::cm_transport;

DocanListenerImpl::DocanListenerImpl(std::function<void(docan_indication*)> indication_callback,
                    std::function<void(docan_confirm*)> confirm_callback)
: DocanListener()
, indication_callback(indication_callback)
, confirm_callback(confirm_callback)
{
}

DocanListenerImpl::~DocanListenerImpl()
{
}

void
DocanListenerImpl::OnUdsResponse(uint16_t sa, uint16_t ta, uint32_t reqId, docan_result_t result, const std::vector<uint8_t>& uds)
{
    docan_confirm confirm;
    confirm.sa = sa;
    confirm.ta = ta;
    confirm.reqId = reqId;
    confirm.result = result;
    confirm.length = uds.size();
    confirm.uds.assign(uds.begin(), uds.end());
    if (nullptr != confirm_callback) {
        confirm_callback(&confirm);
    }
}

void
DocanListenerImpl::OnUdsIndication(uint16_t ta, uint16_t sa, const std::vector<uint8_t>& uds)
{
    docan_indication indication;
    indication.sa = sa;
    indication.ta = ta;
    indication.length = uds.size();
    indication.uds.assign(uds.begin(), uds.end());
    if (nullptr != indication_callback) {
        indication_callback(&indication);
    }
}

void
DocanListenerImpl::onServiceBind(const std::string& name)
{
}

void
DocanListenerImpl::onServiceUnbind(const std::string& name)
{
}

DiagServerTransPortService::DiagServerTransPortService()
: service_docan_(new DocanService())
, listener_docan_impl_(nullptr)
, request_id_docan_(-1)
, service_doip_(new DoIPTransport())
,doipReqChannel_(Doip_Req_Channel::kDefault)
,docanReqChannel_(Docan_Req_Channel::kDefault)
,time_mgr_(std::make_unique<TimerManager>())
{
}

DiagServerTransPortService*
DiagServerTransPortService::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerTransPortService();
        }
    }

    return instance_;
}

void
DiagServerTransPortService::Init()
{
    DG_INFO << "DiagServerTransPortService::Init";
    DiagServerTransPortCM::getInstance()->Init();
    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }
}

void
DiagServerTransPortService::DeInit()
{
    DG_INFO << "DiagServerTransPortService::DeInit";
    request_id_address_doip2docan_.clear();
    current_link_address_list_.clear();
    DiagServerTransPortCM::getInstance()->DeInit();
    if (nullptr != service_docan_) {
        delete service_docan_;
        service_docan_ = nullptr;
    }

    if (nullptr != service_doip_) {
        delete service_doip_;
        service_doip_ = nullptr;
    }

    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
        time_mgr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

bool
DiagServerTransPortService::DoCanStart(std::function<void(docan_indication*)> indication_callback,
                                      std::function<void(docan_confirm*)>  confirm_callback)
{
    DG_INFO << "DiagServerTransPortService::DoCanStart";
    if (nullptr == service_docan_) {
        DG_ERROR << "DiagServerTransPortService::DoCanStart service_docan_ is nullptr.";
        return false;
    }

    listener_docan_impl_ = std::shared_ptr<DocanListener>(new DocanListenerImpl(indication_callback, confirm_callback));
    service_docan_->Init();
    service_docan_->Start();
    service_docan_->registerListener(Docan_Instance, listener_docan_impl_);
    return true;
}

bool
DiagServerTransPortService::DoCanStop()
{
    DG_INFO << "DiagServerTransPortService::DoCanStop";
    if (nullptr == service_docan_) {
        DG_ERROR << "DiagServerTransPortService::DoCanStop service_docan_ is nullptr.";
        return false;
    }

    service_docan_->unregisterListener(Docan_Instance);
    service_docan_->Stop();
    service_docan_->Deinit();
    return true;
}

bool
DiagServerTransPortService::DoCanRequest(const std::string& instance, const DiagServerReqUdsMessage& udsMessage, const bool isDopipToDocan, const uint16_t doipAddress, const bool someipChannel)
{
    DG_DEBUG << "DiagServerTransPortService::DoCanRequest instance: " << instance << " isDopipToDocan: " << static_cast<int>(isDopipToDocan);
    if (nullptr == service_docan_) {
        DG_ERROR << "DiagServerTransPortService::DoCanRequest service_docan_ is nullptr.";
        return false;
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (docanReqChannel_ != Docan_Req_Channel::kDefault)
    {
        DG_ERROR << "Current DoCanRequest is busy ,ignore this req.";
        return false;
    }

    if (someipChannel)
    {
        DG_DEBUG << "In SomeipChannel!";
        docanReqChannel_ = Docan_Req_Channel::kSomeip;
        
    } else {
        DG_DEBUG << "In NotSomeipChannel!";
        docanReqChannel_ = Docan_Req_Channel::kNotSomeip;
    }

    time_fd_docan_ = -1;
    time_mgr_->StartFdTimer(time_fd_docan_, 2000, std::bind(&DiagServerTransPortService::DocanSessionTimeout, this, std::placeholders::_1), NULL, false);
#endif
    int32_t requestId = service_docan_->UdsRequest(instance, udsMessage.udsSa, udsMessage.udsTa, udsMessage.udsData);
    DG_DEBUG << "DiagServerTransPortService::DoCanRequest requestId: " << requestId;
    if (-1 == requestId) {
        DG_ERROR << "DiagServerTransPortService::DoCanRequest DocanListenerImpl regist failed.";
        return false;
    }

    if (isDopipToDocan) {
        request_id_address_doip2docan_.insert(std::make_pair(requestId, doipAddress));
    }
    else {
        request_id_docan_ = requestId;
    }

    return true;
}

bool
DiagServerTransPortService::DoIPStart(std::function<void(doip_indication_t*)> indication_callback,
                             std::function<void(doip_confirm_t*)> confirm_callback,
                             std::function<void(doip_route_t*)> route_callback)
{
    DG_INFO << "DiagServerTransPortService::DoIPStart";
    if (nullptr == service_doip_) {
        DG_ERROR << "DiagServerTransPortService::DoIPStart service_doip_ is nullptr.";
        return false;
    }

    auto initResult = service_doip_->DoipInit(indication_callback, confirm_callback, route_callback);
    if (DOIP_RESULT_OK != initResult) {
        DG_ERROR << "DiagServerTransPortService::DoIPStart DoipInit failed. failcode: " << initResult;
        return false;
    }

    service_doip_->DoipRegistReleaseCallback(std::bind(&DiagServerTransPortService::DoIPLinkStatusCallBack, this, std::placeholders::_1, std::placeholders::_2));
    return true;
}

bool
DiagServerTransPortService::DoIPStop()
{
    DG_INFO << "DiagServerTransPortService::DoIPStop";
    if (nullptr == service_doip_) {
        DG_ERROR << "DiagServerTransPortService::DoIPStop service_doip_ is nullptr.";
        return false;
    }

    service_doip_->DoipDeinit();
    return true;
}

bool
DiagServerTransPortService::DoIPRequestByNode(const DiagServerReqUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransPortService::DoIPRequestByNode";
    if (nullptr == service_doip_) {
        DG_ERROR << "DiagServerTransPortService::DoIPRequestByNode service_doip_ is nullptr.";
        return false;
    }

    doip_request_t request;
    request.logical_source_address = udsMessage.udsSa;
    request.logical_target_address = udsMessage.udsTa;
    request.ta_type = static_cast<DOIP_TA_TYPE>(udsMessage.taType);
    request.data_length = udsMessage.udsData.size();
    request.data = (char*)udsMessage.udsData.data();
    auto requestResult = service_doip_->DoipRequestByNode(&request);
    if (DOIP_RESULT::DOIP_RESULT_OK != requestResult) {
        DG_ERROR << "DiagServerTransPortService::DoIPRequestByNode DoipRequestByNode failed. failcode: " << UINT8_TO_STRING(requestResult);
        return false;
    }

    return true;
}

bool
DiagServerTransPortService::DoipRequestByEquip(const DiagServerReqUdsMessage& udsMessage, const bool someipChannel)
{
    DG_DEBUG << "DiagServerTransPortService::DoipRequestByEquip";
    if (nullptr == service_doip_) {
        DG_ERROR << "DiagServerTransPortService::DoipRequestByEquip service_doip_ is nullptr.";
        return false;
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (doipReqChannel_ != Doip_Req_Channel::kDefault)
    {
        DG_ERROR << "Current DoipRequestByEquip is busy ,ignore this req.";
        return false;
    }

    if (someipChannel)
    {
        DG_DEBUG << "In SomeipChannel!";
        doipReqChannel_ = Doip_Req_Channel::kSomeip;
    } else {
        DG_DEBUG << "In NotSomeipChannel!";
        doipReqChannel_ = Doip_Req_Channel::kNotSomeip;
    }

    time_fd_doip_ = -1;
    time_mgr_->StartFdTimer(time_fd_doip_, 2000, std::bind(&DiagServerTransPortService::DoipSessionTimeout, this, std::placeholders::_1), NULL, false);
#endif
    doip_request_t request;
    request.logical_source_address = udsMessage.udsSa;
    request.logical_target_address = udsMessage.udsTa;
    request.ta_type = static_cast<DOIP_TA_TYPE>(udsMessage.taType);
    request.data_length = udsMessage.udsData.size();
    request.data = (char*)udsMessage.udsData.data();
    auto requestResult = service_doip_->DoipRequestByEquip(&request);
    if (DOIP_RESULT::DOIP_RESULT_OK != requestResult) {
        DG_ERROR << "DiagServerTransPortService::DoipRequestByEquip DoipRequestByEquip failed. failcode: " << UINT8_TO_STRING(requestResult);
        return false;
    }

    return true;
}

void
DiagServerTransPortService::DoipReleaseByEquip(const DiagServerReqUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransPortService::DoipReleaseByEquip";
    if (nullptr == service_doip_) {
        DG_ERROR << "DiagServerTransPortService::DoipReleaseByEquip service_doip_ is nullptr.";
        return;
    }

    doip_request_t request;
    request.logical_source_address = udsMessage.udsSa;
    request.logical_target_address = udsMessage.udsTa;
    request.ta_type = static_cast<DOIP_TA_TYPE>(udsMessage.taType);
    request.data_length = udsMessage.udsData.size();
    request.data = (char*)udsMessage.udsData.data();
    auto requestResult = service_doip_->DoipReleaseByEquip(&request);
    if (DOIP_RESULT::DOIP_RESULT_OK != requestResult) {
        DG_ERROR << "DiagServerTransPortService::DoipReleaseByEquip DoipReleaseByEquip failed. failcode: " << UINT8_TO_STRING(requestResult);
    }
}

void
DiagServerTransPortService::DoIPLinkStatusCallBack(doip_netlink_status_t status, uint16_t address)
{
    DG_INFO << "DiagServerTransPortService::DoIPLinkStatusCallBack status: " << status << " address: " << UINT16_TO_STRING(address);
    DoipNetlinkStatus diag_status;
    if (status == DOIP_NETLINK_STATUS::DOIP_NETLINK_STATUS_UP) {
        diag_status = DoipNetlinkStatus::kUp;
        auto itr = find(current_link_address_list_.begin(), current_link_address_list_.end(), address);
        if (itr == current_link_address_list_.end()) {
            current_link_address_list_.push_back(address);
        }
    }
    else {
        diag_status = DoipNetlinkStatus::kDown;
        auto itr = find(current_link_address_list_.begin(), current_link_address_list_.end(), address);
        if (itr != current_link_address_list_.end()) {
            current_link_address_list_.erase(itr);
        }
    }

    DiagServerTransport::getInstance()->NotifyDoipNetlinkStatus(diag_status, address);
}

bool
DiagServerTransPortService::GetDoipAddressByRequestId(const int32_t& requestId, uint16_t& address)
{
    DG_DEBUG << "DiagServerTransPortService::GetDoipAddress requestId: " << requestId;
    auto itr = request_id_address_doip2docan_.find(requestId);
    if (itr == request_id_address_doip2docan_.end()) {
        return false;
    }

    address = itr->second;
    return true;
}

void
DiagServerTransPortService::DeleteDoipAddressByRequestId(const int32_t& requestId)
{
    DG_DEBUG << "DiagServerTransPortService::DeleteDoipAddressByRequestId requestId: " << requestId;
    auto itr = request_id_address_doip2docan_.find(requestId);
    if (itr == request_id_address_doip2docan_.end()) {
        return;
    }

    request_id_address_doip2docan_.erase(itr);
}

bool 
DiagServerTransPortService::DoSomeIPStart(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback)
{
#ifdef BUILD_SOMEIP_ENABLE
    DG_INFO << "DiagServerTransPortService::DoSomeIPStart";

    auto initResult = DoSomeIPTransport::getInstance()->DosomeipInit(uds_request_callback, someip_register_callback);
    if (DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK != initResult) {
        DG_ERROR << "DiagServerTransPortService::DoSomeIPStart DoipInit failed. failcode: " << static_cast<uint16_t>(initResult);
        return false;
    }
    return true;
#else
    return true;
#endif
}

bool 
DiagServerTransPortService::DoSomeIPStop()
{
#ifdef BUILD_SOMEIP_ENABLE
    DG_INFO << "DiagServerTransPortService::DoSomeIPStop";
    DoSomeIPTransport::getInstance()->DosomeipDeinit();
    return true;
#else
    return true;
#endif
}

bool 
DiagServerTransPortService::ReplyUdsOnSomeIp(const DoSomeIPRespUdsMessage& udsMsg, const Req_Channel& channel)
{
#ifdef BUILD_SOMEIP_ENABLE
    DG_INFO << "DiagServerTransPortService::ReplyUdsOnSomeIp";
    auto initResult = DoSomeIPTransport::getInstance()->ReplyUdsOnSomeIp(udsMsg);
    switch (channel)
    {
    case Req_Channel::kServer:
        // do nothing
        break;
    case Req_Channel::kDocan:
        setDocanChannelEmpty();
        break;
    case Req_Channel::kDoip:
        setDoipChannelEmpty();
        break;
    default:
        break;
    }
    if (DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK != initResult) {
        DG_ERROR << "DiagServerTransPortService::DoSomeIPStart DoipInit failed. failcode: " << static_cast<uint16_t>(initResult);
        return false;
    }
    return true;
#else
    return true;
#endif
}

Docan_Req_Channel 
DiagServerTransPortService::GetDocanChannel()
{
    DG_DEBUG << "GetDocanChannel() : " << static_cast<uint16_t>(docanReqChannel_);
    return docanReqChannel_;
}


Doip_Req_Channel 
DiagServerTransPortService::GetDoipChannel()
{
    DG_DEBUG << "GetDoipChannel() : " << static_cast<uint16_t>(doipReqChannel_);
    return doipReqChannel_;
}

void 
DiagServerTransPortService::setDocanChannelEmpty()
{
    DG_DEBUG << "setDocanChannelEmpty StopFdTimer";
    docanReqChannel_ = Docan_Req_Channel::kDefault;
    time_mgr_->StopFdTimer(time_fd_docan_);
    time_fd_docan_ = -1 ;
}

void
DiagServerTransPortService::setDoipChannelEmpty()
{
    DG_DEBUG << "setDoipChannelEmpty StopFdTimer";
    doipReqChannel_ = Doip_Req_Channel::kDefault;
    time_mgr_->StopFdTimer(time_fd_doip_);
    time_fd_doip_ = -1;
}

void
DiagServerTransPortService::DoipSessionTimeout(void * data)
{
    DG_DEBUG << "DiagServerTransPortService::DoipSessionTimeout switch doipReqChannel_ To Default";
    doipReqChannel_ = Doip_Req_Channel::kDefault;
}

void 
DiagServerTransPortService::DocanSessionTimeout(void * data)
{
    DG_DEBUG << "DiagServerTransPortService::DocanSessionTimeout switch docanReqChannel_ To Default";
    docanReqChannel_ = Docan_Req_Channel::kDefault;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon