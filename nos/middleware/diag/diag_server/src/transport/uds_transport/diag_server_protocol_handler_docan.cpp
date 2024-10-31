/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server protocol handler(DoCAN)
*/

#include <functional>
#include <thread>

#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_docan.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/common/diag_server_config.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

using namespace hozon::netaos::diag::cm_transport;

DoCAN_UdsTransportProtocolHandler::DoCAN_UdsTransportProtocolHandler(const UdsTransportProtocolHandlerID handlerId,
        UdsTransportProtocolMgr& transportProtocolMgr)
: UdsTransportProtocolHandler(handlerId, transportProtocolMgr)
{
}

DoCAN_UdsTransportProtocolHandler::~DoCAN_UdsTransportProtocolHandler()
{
}

DoCAN_UdsTransportProtocolHandler::InitializationResult
DoCAN_UdsTransportProtocolHandler::Initialize() const
{
    DG_INFO << "DoCAN_UdsTransportProtocolHandler::Initialize.";
    return InitializationResult::INITIALIZE_OK;
}

bool
DoCAN_UdsTransportProtocolHandler::NotifyReestablishment() const
{
    return true;
}

bool
DoCAN_UdsTransportProtocolHandler::Start()
{
    DG_INFO << "DoCAN_UdsTransportProtocolHandler::Start.";
    bool bResult = DiagServerTransPortService::getInstance()->DoCanStart(
                                std::bind(&DoCAN_UdsTransportProtocolHandler::DoCanIndicationCallback, this, std::placeholders::_1),
                                std::bind(&DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback, this, std::placeholders::_1));
    return bResult;
}

void
DoCAN_UdsTransportProtocolHandler::Stop()
{
    DG_INFO << "DoCAN_UdsTransportProtocolHandler::Stop handlerId_: " << GetHandlerId() << ".";
    DiagServerTransPortService::getInstance()->DoCanStop();
    transportProtocolManager_.HandlerStopped(GetHandlerId());
}

void
DoCAN_UdsTransportProtocolHandler::Transmit(UdsMessageConstPtr message, ChannelID transmitChannelId) const
{
    DG_DEBUG << "DoCAN_UdsTransportProtocolHandler::Transmit transmitChannelId: " << transmitChannelId << ".";
    if ((nullptr == message) || (0 == message->GetUdsPayload().size())) {
        DG_ERROR << "DoCAN_UdsTransportProtocolHandler::Transmit message is null.";
        return;
    }

    // TO DO send uds response to can
    DiagServerReqUdsMessage udsmessage;
    DiagServerTransPortService::getInstance()->DoCanRequest(Docan_Instance, udsmessage);

    // TO DO wait confirm result from can
    transportProtocolManager_.TransmitConfirmation(std::move(message), UdsTransportProtocolMgr::TransmissionResult::TRANSMIT_OK);
}

void
DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback(docan_confirm* docanConfirm)
{
    DG_DEBUG << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback result: " << docanConfirm->result
                                                                     << " reqid: " << docanConfirm->reqId;
    if ((nullptr == docanConfirm) || (0 == docanConfirm->length)) {
        DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback docanConfirm is nullptr or data length is 0.";
        return;
    }

    if (docan_result_t::OK != docanConfirm->result) {
        DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback result failcode: " << docanConfirm->result;
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (DiagServerTransPortService::getInstance()->GetDocanChannel() == Docan_Req_Channel::kNotSomeip)
    {
        DiagServerTransPortService::getInstance()->setDocanChannelEmpty();
    }
#endif
    // docan to doip
    uint16_t address = 0x00;
    bool bResult = DiagServerTransPortService::getInstance()->GetDoipAddressByRequestId(static_cast<int32_t>(docanConfirm->reqId), address);
    if (bResult) {
        if (address != docanConfirm->ta) {
            DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback doip adress: " << UINT16_TO_STRING(address)
                                                                       << " docanConfirm->ta: " << UINT16_TO_STRING(docanConfirm->ta) << " not match. ";
            return;
        }

        DiagServerReqUdsMessage udsMessage;
        udsMessage.udsSa = docanConfirm->sa;
        udsMessage.udsTa = docanConfirm->ta;
        udsMessage.busType = DiagUdsBusType::kDoip;
        udsMessage.taType = DiagTargetAddressType::kPhysical;
        udsMessage.udsData.assign(docanConfirm->uds.begin(), docanConfirm->uds.end());
        DG_INFO << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                       << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
                                                                       << " udsdata.size: " << udsMessage.udsData.size()
                                                                       << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);

        auto requestResult = DiagServerTransPortService::getInstance()->DoIPRequestByNode(udsMessage);
        if (!requestResult) {
            DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback DoIPRequestByNode failed.";
        }

        DiagServerTransPortService::getInstance()->DeleteDoipAddressByRequestId(static_cast<int32_t>(docanConfirm->reqId));
        return;
    }

    // docan to update manager or remote
    if (DiagServerConfig::getInstance()->IsUpdateManagerOrRemoteAddress(docanConfirm->ta)) {
        // int32_t requestId = DiagServerTransPortService::getInstance()->GetDocanRequestId();
        // if (requestId != static_cast<int32_t>(docanConfirm->reqId)) {
        //     DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback error reqid: " << docanConfirm->reqId;
        //     return;
        // }

        DiagServerRespUdsMessage udsmessage;
        udsmessage.udsSa = docanConfirm->sa;
        udsmessage.udsTa = docanConfirm->ta;
        udsmessage.busType = DiagUdsBusType::kDocan;
        udsmessage.result = docanConfirm->result;
        udsmessage.udsData.assign(docanConfirm->uds.begin(), docanConfirm->uds.end());
        DiagServerTransPortCM::getInstance()->DiagEventSend(udsmessage, DiagServerConfig::getInstance()->IsRemoteAddress(docanConfirm->ta));
        DiagServerTransPortService::getInstance()->DeleteDoipAddressByRequestId(static_cast<int32_t>(docanConfirm->reqId));
        return;
    }
#ifdef BUILD_SOMEIP_ENABLE
    // 如果目标地址是someip通道,则主动将消息发送回去，塞回someip通道
    if (DoSomeIPConfig::Instance()->IsDoSomeipProxyAddress(docanConfirm->ta) && DiagServerTransPortService::getInstance()->GetDocanChannel() == Docan_Req_Channel::kSomeip) 
    {
        DG_DEBUG << "docan ta is dosomeIp!";
        DoSomeIPRespUdsMessage udsmessage{};
        udsmessage.udsSa = docanConfirm->sa;
        udsmessage.udsTa = docanConfirm->ta;
        udsmessage.result = static_cast<uint16_t>(docanConfirm->result);
        udsmessage.udsData.assign(docanConfirm->uds.begin(), docanConfirm->uds.end());
        DiagServerTransPortService::getInstance()->ReplyUdsOnSomeIp(udsmessage, Req_Channel::kDocan);
        DiagServerTransPortService::getInstance()->DeleteDoipAddressByRequestId(static_cast<int32_t>(docanConfirm->reqId));
        return;
    }
#endif

    int32_t requestId = DiagServerTransPortService::getInstance()->GetDocanRequestId();
    if (requestId != static_cast<int32_t>(docanConfirm->reqId)) {
        DG_ERROR << "DoCAN_UdsTransportProtocolHandler::DoCanConfirmCallback error reqid: " << docanConfirm->reqId;
        return;
    }

    // docan
    DiagServerTransPortService::getInstance()->DeleteDocanRequestId();
}

void
DoCAN_UdsTransportProtocolHandler::DoCanIndicationCallback(docan_indication* docanIndication)
{
    DG_DEBUG << "DoCAN_UdsTransportProtocolHandler::DoCanIndicationCallback.";
}

}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon