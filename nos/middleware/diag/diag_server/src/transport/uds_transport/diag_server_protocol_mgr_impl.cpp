/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server protocol mgr impl
*/

#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_mgr_impl.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_def.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

UdsTransportProtocolMgrImpl::UdsTransportProtocolMgrImpl()
{
}

UdsTransportProtocolMgrImpl::~UdsTransportProtocolMgrImpl()
{

}

void
UdsTransportProtocolMgrImpl::ChannelReestablished(GlobalChannelIdentifier globalChannelId) const
{

}

void
UdsTransportProtocolMgrImpl::HandleMessage(UdsMessagePtr message) const
{
    DG_DEBUG << "UdsTransportProtocolMgrImpl::HandleMessage enter.";
    if ((nullptr == message) || (message->GetUdsPayload().empty())) {
        DG_ERROR << "UdsTransportProtocolMgrImpl::HandleMessage message is nullptr or uds payload is empty.";
        return;
    }

    DiagServerUdsMessage udsMessage;
    udsMessage.udsSa = message->GetSa();
    udsMessage.udsTa = message->GetTa();
    udsMessage.taType = static_cast<DiagTargetAddressType>(message->GetTaType());
    ByteVector payLoad = message->GetUdsPayload();
    for (uint32_t i = 0; i < payLoad.size(); i++) {
        udsMessage.udsData.push_back(payLoad[i]);
    }

    DG_INFO << "UdsTransportProtocolMgrImpl::HandleMessage sa: " << UINT16_TO_STRING(udsMessage.udsSa)
            << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
            << " udsdata.size: " << udsMessage.udsData.size()
            << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    DiagServerTransport::getInstance()->RecvUdsMessage(udsMessage);
}

void
UdsTransportProtocolMgrImpl::HandlerStopped(UdsTransportProtocolHandlerID handlerId) const
{
    DG_DEBUG << "UdsTransportProtocolMgrImpl::HandlerStopped handlerId: " << handlerId << ".";
    DiagServerTransport::getInstance()->HandlerStopped(handlerId);
}

std::pair<UdsTransportProtocolMgr::IndicationResult, UdsMessagePtr>
UdsTransportProtocolMgrImpl::IndicateMessage(UdsMessage::Address sourceAddr,
                                                    UdsMessage::Address targetAddr,
                                                    UdsMessage::TargetAddressType type,
                                                    GlobalChannelIdentifier globalChannelId,
                                                    std::size_t size, Priority priority, ProtocolKind protocolKind,
                                                    std::vector<uint8_t>& payloadInfo) const
{
    // TODO Conditional judgement(unknownTA overflow busy)
    UdsMessagePtr uds_msg_ptr_ = nullptr;
    std::pair<UdsTransportProtocolMgr::IndicationResult, UdsMessagePtr> result;
    bool isUnknownTA = false;
    if (isUnknownTA) {
        result.first = IndicationResult::INDICATION_UNKNOWN_TARGET_ADDRESS;
        result.second = std::move(uds_msg_ptr_);
        return result;
    }

    bool isOverflow = false;
    if (isOverflow) {
        result.first = IndicationResult::INDICATION_OVERFLOW;
        result.second = std::move(uds_msg_ptr_);
        return result;
    }

    bool isBusy = false;
    if (isBusy) {
        result.first = IndicationResult::INDICATION_BUSY;
        result.second = std::move(uds_msg_ptr_);
        return result;
    }

    const ByteVector payload(payloadInfo.begin(), payloadInfo.end());
    uds_msg_ptr_ = std::make_unique<UdsMessageImpl>(sourceAddr, targetAddr, type, payload);
    result.first = IndicationResult::INDICATION_OK;
    result.second = std::move(uds_msg_ptr_);

    return result;
}

void
UdsTransportProtocolMgrImpl::NotifyMessageFailure(UdsMessagePtr message) const
{
    DG_DEBUG << "UdsTransportProtocolMgrImpl::NotifyMessageFailure enter.";
    if ((nullptr == message) || (message->GetUdsPayload().empty())) {
        DG_ERROR << "UdsTransportProtocolMgrImpl::NotifyMessageFailure message is nullptr or uds payload is empty.";
        return;
    }

    DiagServerUdsMessage udsMessage;
    udsMessage.udsSa = message->GetSa();
    udsMessage.udsTa = message->GetTa();
    ByteVector payLoad = message->GetUdsPayload();
    for (uint32_t i = 0; i < payLoad.size(); i++) {
        udsMessage.udsData.push_back(payLoad[i]);
    }

    DG_INFO << "UdsTransportProtocolMgrImpl::NotifyMessageFailure sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                             << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
                                                             << " udsdata.size: " << udsMessage.udsData.size()
                                                             << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    DiagServerTransport::getInstance()->NotifyMessageFailure(udsMessage);
}

void
UdsTransportProtocolMgrImpl::TransmitConfirmation(UdsMessageConstPtr message, TransmissionResult result) const
{
    // DG_DEBUG << "UdsTransportProtocolMgrImpl::TransmitConfirmation enter.";
    if ((nullptr == message) || (message->GetUdsPayload().empty())) {
        DG_ERROR << "UdsTransportProtocolMgrImpl::TransmitConfirmation message is nullptr or uds payload is empty.";
        return;
    }

    DiagServerUdsMessage udsMessage;
    udsMessage.udsSa = message->GetSa();
    udsMessage.udsTa = message->GetTa();
    ByteVector payLoad = message->GetUdsPayload();
    for (uint32_t i = 0; i < payLoad.size(); i++) {
        udsMessage.udsData.push_back(payLoad[i]);
    }

    // DG_INFO << "UdsTransportProtocolMgrImpl::TransmitConfirmation sa: " << UINT16_TO_STRING(udsMessage.udsSa)
    //                                                          << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
    //                                                          << " udsdata.size: " << udsMessage.udsData.size()
    //                                                          << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    bool bConfirmResult = false;
    if (TransmissionResult::TRANSMIT_OK == result) {
        bConfirmResult = true;
    }

    DiagServerTransport::getInstance()->TransmitConfirmation(udsMessage, bConfirmResult);
}

}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
