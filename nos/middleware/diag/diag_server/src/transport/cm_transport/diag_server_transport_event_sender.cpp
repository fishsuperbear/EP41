/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: tpl event sender
*/

#include "idl/generated/diagPubSubTypes.h"
#include "diag_server_transport_event_sender.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {


DiagServerTransportEventSender::DiagServerTransportEventSender()
: diag_skeleton_(nullptr)
, remote_diag_skeleton_(nullptr)
, diag_session_skeleton_(nullptr)
{
}

DiagServerTransportEventSender::~DiagServerTransportEventSender()
{
}

void
DiagServerTransportEventSender::Init()
{
    DG_INFO << "DiagServerTransportEventSender::Init";
    // diag skeleton init
    std::shared_ptr<uds_raw_data_resp_eventPubSubType> diagPubsubtype = std::make_shared<uds_raw_data_resp_eventPubSubType>();
    diag_skeleton_ = std::make_shared<Skeleton>(diagPubsubtype);
    diag_skeleton_->Init(0, "uds_raw_data_resp_eventTopic");

    // remote diag skeleton init
    std::shared_ptr<remote_uds_raw_data_resp_eventPubSubType> remoteDiagPubsubtype = std::make_shared<remote_uds_raw_data_resp_eventPubSubType>();
    remote_diag_skeleton_ = std::make_shared<Skeleton>(remoteDiagPubsubtype);
    remote_diag_skeleton_->Init(0, "remote_uds_raw_data_resp_eventTopic");

    // diag session skeleton init
    std::shared_ptr<uds_current_session_notify_eventPubSubType> diagSessionPubsubtype = std::make_shared<uds_current_session_notify_eventPubSubType>();
    diag_session_skeleton_ = std::make_shared<Skeleton>(diagSessionPubsubtype);
    diag_session_skeleton_->Init(0, "uds_current_session_notify_eventTopic");
}

void
DiagServerTransportEventSender::DeInit()
{
    DG_INFO << "DiagServerTransportEventSender::DeInit";
    // diag session skeleton deinit
    if (nullptr != diag_session_skeleton_) {
        diag_session_skeleton_->Deinit();
        diag_session_skeleton_ = nullptr;
    }

    // remote diag skeleton deinit
    if (nullptr != remote_diag_skeleton_) {
        remote_diag_skeleton_->Deinit();
        remote_diag_skeleton_ = nullptr;
    }

    // diag skeleton deinit
    if (nullptr != diag_skeleton_) {
        diag_skeleton_->Deinit();
        diag_skeleton_ = nullptr;
    }
}

void
DiagServerTransportEventSender::DiagEventSend(const DiagServerRespUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransportEventSender::DiagEventSend busType: " << udsMessage.busType << ".";
    if (nullptr == diag_skeleton_) {
        DG_ERROR << "DiagServerTransportEventSender::DiagEventSend diag_skeleton_ is nullptr.";
        return;
    }

    std::shared_ptr<uds_raw_data_resp_event> data = std::make_shared<uds_raw_data_resp_event>();
    data->sa(udsMessage.udsSa);
    data->ta(udsMessage.udsTa);
    data->bus_type(udsMessage.busType);
    data->result(udsMessage.result);
    data->data_vec(udsMessage.udsData);

    if (diag_skeleton_->IsMatched()) {
        if (0 == diag_skeleton_->Write(data)) {
            DG_INFO << "DiagServerTransportEventSender::DiagEventSend -> sa: " << UINT16_TO_STRING(data->sa()) \
                    << " ta: " << UINT16_TO_STRING(data->ta()) << " bus_type: " << data->bus_type() \
                    << " result: " << data->result() << " data_vec: " << UINT8_VEC_TO_STRING(data->data_vec());
        }
    }
}

void
DiagServerTransportEventSender::RemoteDiagEventSend(const DiagServerRespUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransportEventSender::RemoteDiagEventSend busType: " << udsMessage.busType << ".";
    if (nullptr == remote_diag_skeleton_) {
        DG_ERROR << "DiagServerTransportEventSender::RemoteDiagEventSend remote_skeleton_ is nullptr.";
        return;
    }

    std::shared_ptr<remote_uds_raw_data_resp_event> data = std::make_shared<remote_uds_raw_data_resp_event>();
    data->sa(udsMessage.udsSa);
    data->ta(udsMessage.udsTa);
    data->bus_type(udsMessage.busType);
    data->result(udsMessage.result);
    data->data_vec(udsMessage.udsData);

    if (remote_diag_skeleton_->IsMatched()) {
        if (0 == remote_diag_skeleton_->Write(data)) {
            DG_INFO << "DiagServerTransportEventSender::RemoteDiagEventSend -> sa: " << UINT16_TO_STRING(data->sa()) \
                    << " ta: " << UINT16_TO_STRING(data->ta()) << " bus_type: " << data->bus_type() \
                    << " result: " << data->result() << " data_vec: " << UINT8_VEC_TO_STRING(data->data_vec());
        }
    }
}

void
DiagServerTransportEventSender::DiagSessionEventSend(const DiagServerSessionCode& session)
{
    DG_DEBUG << "DiagServerTransportEventSender::DiagSessionEventSend session: " << session << ".";
    if (nullptr == diag_session_skeleton_) {
        DG_ERROR << "DiagServerTransportEventSender::DiagSessionEventSend diag_session_skeleton_ is nullptr.";
        return;
    }

    std::shared_ptr<uds_current_session_notify_event> data = std::make_shared<uds_current_session_notify_event>();
    data->current_session(session);

    if (diag_session_skeleton_->IsMatched()) {
        if (0 == diag_session_skeleton_->Write(data)) {
            DG_INFO << "DiagServerTransportEventSender::DiagSessionEventSend -> current_session: " << UINT8_TO_STRING(data->current_session());
        }
    }
}

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon