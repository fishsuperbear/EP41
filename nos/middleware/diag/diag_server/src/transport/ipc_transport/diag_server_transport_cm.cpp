/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server cm
*/

#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/info/diag_server_chassis_info.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

#define UDS_DATA_TASK_THREAD_NUM        10

DiagServerTransPortCM* DiagServerTransPortCM::instance_ = nullptr;
std::mutex DiagServerTransPortCM::mtx_;

DiagServerTransPortCM::DiagServerTransPortCM()
: event_sender_(new DiagServerTransportEventSender())
, event_receiver_(new DiagServerTransportEventReceiver())
, method_sender_(new DiagServerTransportMethodSender())
, method_receiver_(new DiagServerTransportMethodReceiver())
{
}

DiagServerTransPortCM*
DiagServerTransPortCM::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerTransPortCM();
        }
    }

    return instance_;
}

void
DiagServerTransPortCM::Init()
{
    DG_INFO << "DiagServerTransPortCM::Init";
    threadpool_ = std::make_unique<ThreadPool>(UDS_DATA_TASK_THREAD_NUM);
    // event receiver init
    if (event_receiver_ != nullptr) {
        event_receiver_->Init();
    }

    // event dispatcher init
    if (event_sender_ != nullptr) {
        event_sender_->Init();
    }

    // method receiver init
    if (method_receiver_ != nullptr) {
        method_receiver_->Init();
    }

    // method dispatcher init
    if (method_sender_ != nullptr) {
        method_sender_->Init();
    }
}

void
DiagServerTransPortCM::DeInit()
{
    DG_INFO << "DiagServerTransPortCM::DeInit";
    threadpool_->StopAll();
    // event dispatcher deinit
    if (event_sender_ != nullptr) {
        event_sender_->DeInit();
        delete event_sender_;
        event_sender_ = nullptr;
    }

    // event receiver deinit
    if (event_receiver_ != nullptr) {
        event_receiver_->DeInit();
        delete event_receiver_;
        event_receiver_ = nullptr;
    }

    // method dispatcher deinit
    if (method_sender_ != nullptr) {
        method_sender_->DeInit();
        delete method_sender_;
        method_sender_ = nullptr;
    }

    // method receiver deinit
    if (method_receiver_ != nullptr) {
        method_receiver_->DeInit();
        delete method_receiver_;
        method_receiver_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

// event
void
DiagServerTransPortCM::DiagEventSend(const DiagServerRespUdsMessage& udsMessage, const bool remote)
{
    DG_DEBUG << "DiagServerTransPortCM::DiagEventSend busType: " << udsMessage.busType << " remote: " << static_cast<uint>(remote) << ".";
    if (nullptr == event_sender_) {
        DG_ERROR << "DiagServerTransPortCM::DiagEventSend event_sender_ is nullptr.";
        return;
    }

    if (remote) {
        event_sender_->RemoteDiagEventSend(udsMessage);
    }
    else {
        event_sender_->DiagEventSend(udsMessage);
    }
}

void
DiagServerTransPortCM::DiagEventCallback(const DiagServerReqUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransPortCM::DiagEventCallback busType: " << udsMessage.busType << ".";
    if (DiagUdsBusType::kDoip_DisConn == udsMessage.busType) {
        DiagServerTransPortService::getInstance()->DoipReleaseByEquip(udsMessage);
        return;
    }

    DiagServerReqUdsMessage* udsMessagePtr = new DiagServerReqUdsMessage;
    udsMessagePtr->busType = udsMessage.busType;
    udsMessagePtr->udsSa = udsMessage.udsSa;
    udsMessagePtr->udsTa = udsMessage.udsTa;
    udsMessagePtr->taType = udsMessage.taType;
    udsMessagePtr->udsData.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());

    UdsDataTask *task = new UdsDataTask("UDS_REQUEST");
    task->setData((void*)(udsMessagePtr));
    threadpool_->AddTask(task);
}

void
DiagServerTransPortCM::DiagSessionEventSend(const DiagServerSessionCode& session)
{
    DG_DEBUG << "DiagServerTransPortCM::DiagSessionEventSend session: " << session << ".";
    if (nullptr == event_sender_) {
        DG_ERROR << "DiagServerTransPortCM::DiagSessionEventSend event_sender_ is nullptr.";
        return;
    }

    event_sender_->DiagSessionEventSend(session);
}

// method
void
DiagServerTransPortCM::DiagMethodSend(const uint8_t sid, const uint8_t subFunc, const std::vector<std::string> service, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerTransPortCM::DiagMethodSend sid: " << UINT8_TO_STRING(sid) << " subFunc: " << UINT8_TO_STRING(subFunc) << " service.size: " << service.size()
                                                          << " udsdata.size: " << udsData.size() <<  " udsdata: " << UINT8_VEC_TO_STRING(udsData);
    if (nullptr == method_sender_) {
        DG_ERROR << "DiagServerTransPortCM::DiagMethodSend method_sender_ is nullptr.";
        return;
    }

    method_sender_->DiagMethodSend(sid, subFunc, service, udsData);
}

void
DiagServerTransPortCM::ChassisMethodSend()
{
    DG_DEBUG << "DiagServerTransPortCM::ChassisMethodSend.";
    if (nullptr == method_sender_) {
        DG_ERROR << "DiagServerTransPortCM::ChassisMethodSend method_sender_ is nullptr.";
        return;
    }

    method_sender_->ChassisMethodSend();
}

bool
DiagServerTransPortCM::IsCheckUpdateStatusOk()
{
    DG_DEBUG << "DiagServerTransPortCM::IsCheckUpdateStatusOk.";
    if (nullptr == method_sender_) {
        DG_ERROR << "DiagServerTransPortCM::IsCheckUpdateStatusOk method_sender_ is nullptr.";
        return false;
    }

    return method_sender_->IsCheckUpdateStatusOk();
}

UdsDataTask::~UdsDataTask() {
    if (nullptr != ptrData_) {
        delete (DiagServerReqUdsMessage*)ptrData_;
        ptrData_ = nullptr;
    }
}

int
UdsDataTask::Run()
{
    DG_DEBUG << "UdsDataTask::Run.";
    DiagServerReqUdsMessage* udsMessagePtr = (DiagServerReqUdsMessage*)ptrData_;
    DiagServerReqUdsMessage udsMessage;
    udsMessage.busType = udsMessagePtr->busType;
    udsMessage.udsSa = udsMessagePtr->udsSa;
    udsMessage.udsTa = udsMessagePtr->udsTa;
    udsMessage.taType = udsMessagePtr->taType;
    udsMessage.udsData.assign(udsMessagePtr->udsData.begin(), udsMessagePtr->udsData.end());

    bool bResult = false;
    if (DiagUdsBusType::kDocan == udsMessage.busType) {
        bResult = DiagServerTransPortService::getInstance()->DoCanRequest(Docan_Instance, udsMessage);
    }
    else if (DiagUdsBusType::kDoip == udsMessage.busType) {
        bResult = DiagServerTransPortService::getInstance()->DoipRequestByEquip(udsMessage);
    }
    else if (DiagUdsBusType::kServer == udsMessage.busType) {
        DiagServerUdsMessage udsMessage;
        udsMessage.udsSa = udsMessagePtr->udsSa;
        udsMessage.udsTa = udsMessagePtr->udsTa;
        udsMessage.taType = udsMessagePtr->taType;
        udsMessage.udsData.assign(udsMessagePtr->udsData.begin(), udsMessagePtr->udsData.end());
        DiagServerTransport::getInstance()->RecvUdsMessage(udsMessage);
        bResult = true;
    }

    DG_DEBUG << "UdsDataTask::Run request bResult: " << static_cast<int>(bResult) << ".";
    return 0;
}

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon