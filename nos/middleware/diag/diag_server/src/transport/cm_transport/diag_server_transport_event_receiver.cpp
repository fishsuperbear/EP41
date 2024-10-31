/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: tpl event receiver
*/

#include "idl/generated/diagPubSubTypes.h"
#include "diag_server_transport_event_receiver.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {


DiagServerTransportEventReceiver::DiagServerTransportEventReceiver()
: diag_proxy_(nullptr)
, remote_diag_proxy_(nullptr)
{
}

DiagServerTransportEventReceiver::~DiagServerTransportEventReceiver()
{
}

void
DiagServerTransportEventReceiver::Init()
{
    DG_INFO << "DiagServerTransportEventReceiver::Init";
    // diag proxy init
    std::shared_ptr<uds_raw_data_req_eventPubSubType> diagPubsubtype = std::make_shared<uds_raw_data_req_eventPubSubType>();
    diag_proxy_ = std::make_shared<Proxy>(diagPubsubtype);
    diag_proxy_->Init(0, "uds_raw_data_req_eventTopic");
    diag_proxy_->Listen(std::bind(&DiagServerTransportEventReceiver::DiagEventCallback, this));

    // remote diag proxy init
    std::shared_ptr<remote_uds_raw_data_req_eventPubSubType> remoteDiagPubsubtype = std::make_shared<remote_uds_raw_data_req_eventPubSubType>();
    remote_diag_proxy_ = std::make_shared<Proxy>(remoteDiagPubsubtype);
    remote_diag_proxy_->Init(0, "remote_uds_raw_data_req_eventTopic");
    remote_diag_proxy_->Listen(std::bind(&DiagServerTransportEventReceiver::RemoteDiagEventCallback, this));
}

void
DiagServerTransportEventReceiver::DeInit()
{
    DG_INFO << "DiagServerTransportEventReceiver::DeInit";
    // remote diag proxy deinit
    if (nullptr != remote_diag_proxy_) {
        remote_diag_proxy_->Deinit();
        remote_diag_proxy_ = nullptr;
    }

    // diag proxy deinit
    if (nullptr != diag_proxy_) {
        diag_proxy_->Deinit();
        diag_proxy_ = nullptr;
    }
}

void
DiagServerTransportEventReceiver::DiagEventCallback()
{
    DG_DEBUG << "DiagServerTransportEventReceiver::DiagEventCallback.";
    if (nullptr == diag_proxy_) {
        DG_ERROR << "DiagServerTransportEventReceiver::DiagEventCallback diag_proxy_ is nullptr.";
        return;
    }

    if (diag_proxy_->IsMatched()) {
        std::shared_ptr<uds_raw_data_req_event> data = std::make_shared<uds_raw_data_req_event>();
        diag_proxy_->Take(data);
        DG_INFO << "DiagServerTransportEventReceiver::DiagEventCallback -> sa: " << UINT16_TO_STRING(data->sa()) \
                << " ta: " << UINT16_TO_STRING(data->ta()) << " bus_type: " << data->bus_type() \
                << " data_vec: " << UINT8_VEC_TO_STRING(data->data_vec());

        DiagServerReqUdsMessage udsmessage;
        udsmessage.udsSa = data->sa();
        udsmessage.udsTa = data->ta();
        udsmessage.busType = static_cast<DiagUdsBusType>(data->bus_type());
        udsmessage.taType = DiagTargetAddressType::kPhysical;
        udsmessage.udsData.assign(data->data_vec().begin(), data->data_vec().end());
        DiagServerTransPortCM::getInstance()->DiagEventCallback(udsmessage);
    }
}

void
DiagServerTransportEventReceiver::RemoteDiagEventCallback()
{
    DG_DEBUG << "DiagServerTransportEventReceiver::RemoteDiagEventCallback.";
    if (nullptr == remote_diag_proxy_) {
        DG_ERROR << "DiagServerTransportEventReceiver::RemoteDiagEventCallback remote_diag_proxy_ is nullptr.";
        return;
    }

    if (remote_diag_proxy_->IsMatched()) {
        std::shared_ptr<remote_uds_raw_data_req_event> data = std::make_shared<remote_uds_raw_data_req_event>();
        remote_diag_proxy_->Take(data);
        DG_INFO << "DiagServerTransportEventReceiver::RemoteDiagEventCallback -> sa: " << UINT16_TO_STRING(data->sa()) \
                << " ta: " << UINT16_TO_STRING(data->ta()) << " bus_type: " << data->bus_type() \
                << " data_vec: " << UINT8_VEC_TO_STRING(data->data_vec());

        if (DiagServerTransPortService::getInstance()->IsDoipConnecting()) {
            DG_WARN << "DiagServerTransportEventReceiver::RemoteDiagEventCallback doip is connecting, refuse remote diagnostic.";
            DiagServerRespUdsMessage udsMessage;
            udsMessage.udsSa = data->ta();
            udsMessage.udsTa = data->sa();
            udsMessage.busType = static_cast<DiagUdsBusType>(data->bus_type());
            udsMessage.taType = DiagTargetAddressType::kPhysical;
            udsMessage.udsData.push_back(DiagServerNrcErrc::kNegativeHead);
            udsMessage.udsData.push_back(data->data_vec()[0]);
            udsMessage.udsData.push_back(0x0f);
            DiagServerTransPortCM::getInstance()->DiagEventSend(udsMessage, true);
            return;
        }

        DiagServerReqUdsMessage udsmessage;
        udsmessage.udsSa = data->sa();
        udsmessage.udsTa = data->ta();
        udsmessage.busType = static_cast<DiagUdsBusType>(data->bus_type());
        udsmessage.taType = DiagTargetAddressType::kPhysical;
        udsmessage.udsData.assign(data->data_vec().begin(), data->data_vec().end());
        DiagServerTransPortCM::getInstance()->DiagEventCallback(udsmessage);
    }
}

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon