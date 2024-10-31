/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server protocol handler(DoIP)
*/

#include <functional>
#include <thread>

#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_doip.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/transport/diag_server_transport_service.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_config.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

using namespace hozon::netaos::diag::cm_transport;

DoIP_UdsTransportProtocolHandler::DoIP_UdsTransportProtocolHandler(const UdsTransportProtocolHandlerID handlerId,
        UdsTransportProtocolMgr& transportProtocolMgr)
: UdsTransportProtocolHandler(handlerId, transportProtocolMgr)
, doip_confirm_result_(DoIPConfirmResult::CONFIRM_TIMEOUT)
{
}

DoIP_UdsTransportProtocolHandler::~DoIP_UdsTransportProtocolHandler()
{
}

DoIP_UdsTransportProtocolHandler::InitializationResult
DoIP_UdsTransportProtocolHandler::Initialize() const
{
    DG_INFO << "DoIP_UdsTransportProtocolHandler::Initialize.";
    return InitializationResult::INITIALIZE_OK;
}

bool
DoIP_UdsTransportProtocolHandler::NotifyReestablishment() const
{
    return true;
}

bool
DoIP_UdsTransportProtocolHandler::Start()
{
    DG_INFO << "DoIP_UdsTransportProtocolHandler::Start.";
    bool bResult = DiagServerTransPortService::getInstance()->DoIPStart(
                std::bind(&DoIP_UdsTransportProtocolHandler::DoipIndicationCallback, this, std::placeholders::_1),
                std::bind(&DoIP_UdsTransportProtocolHandler::DoipConfirmCallback, this, std::placeholders::_1),
                std::bind(&DoIP_UdsTransportProtocolHandler::DoipRouteCallback, this, std::placeholders::_1));
    return bResult;
}

void
DoIP_UdsTransportProtocolHandler::Stop()
{
    DG_INFO << "DoIP_UdsTransportProtocolHandler::Stop handlerId_: " << GetHandlerId() << ".";
    DiagServerTransPortService::getInstance()->DoIPStop();
    transportProtocolManager_.HandlerStopped(GetHandlerId());
}

void
DoIP_UdsTransportProtocolHandler::Transmit(UdsMessageConstPtr message, ChannelID transmitChannelId) const
{
    DG_DEBUG << "DoIP_UdsTransportProtocolHandler::Transmit transmitChannelId: " << transmitChannelId << ".";
    if ((nullptr == message) || (0 == message->GetUdsPayload().size())) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::Transmit message is null.";
        return;
    }

    DiagServerReqUdsMessage udsMessage;
    udsMessage.udsSa = message->GetSa();
    udsMessage.udsTa = message->GetTa();
    udsMessage.busType = DiagUdsBusType::kDoip;
    udsMessage.taType = static_cast<DiagTargetAddressType>(message->GetTaType());
    ByteVector udsData = message->GetUdsPayload();
    udsMessage.udsData.assign(udsData.begin(), udsData.end());
    DG_INFO << "DoIP_UdsTransportProtocolHandler::Transmit sa: " << UINT16_TO_STRING(udsMessage.udsSa)
            << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
            << " udsdata.size: " << udsMessage.udsData.size()
            << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);

    auto requestResult = DiagServerTransPortService::getInstance()->DoIPRequestByNode(udsMessage);
    if (!requestResult) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::Transmit DoIPRequestByNode failed.";
    }

    // TO DO(temporal strategy)
    {
        int iCount = 0;
        while(DoIPConfirmResult::CONFIRM_TIMEOUT == doip_confirm_result_) {
            if (iCount >= 10) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            iCount++;
        }
    }

    UdsTransportProtocolMgr::TransmissionResult transmissionResult = UdsTransportProtocolMgr::TransmissionResult::TRANSMIT_FAILED;
    if (DoIPConfirmResult::CONFIRM_OK == doip_confirm_result_) {
        transmissionResult = UdsTransportProtocolMgr::TransmissionResult::TRANSMIT_OK;
    }

    DG_INFO << "DoIP_UdsTransportProtocolHandler::Transmit doip_confirm_result_: " << doip_confirm_result_ << " transmissionResult: " << transmissionResult << ".";
    transportProtocolManager_.TransmitConfirmation(std::move(message), transmissionResult);
}

void
DoIP_UdsTransportProtocolHandler::DoipConfirmCallback(doip_confirm_t* doipConfirm)
{
    DG_DEBUG << "DoIP_UdsTransportProtocolHandler::DoipConfirmCallback ta_type " << doipConfirm->ta_type
             << ", result code: " << UINT8_TO_STRING(doipConfirm->result)
             << ", sa: " << UINT16_TO_STRING(doipConfirm->logical_source_address)
             << ", ta: " << UINT16_TO_STRING(doipConfirm->logical_target_address);

    if (nullptr == doipConfirm) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipConfirmCallback doipConfirm is nullptr.";
        return;
    }

    if (DOIP_RESULT::DOIP_RESULT_BUSY == static_cast<uint8_t>(doipConfirm->result)) {
        DG_INFO << "DoIP_UdsTransportProtocolHandler::DoipConfirmCallback result doip is busy.";
    }
    else {
        if (DOIP_RESULT::DOIP_RESULT_OK != doipConfirm->result) {
            DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipConfirmCallback result error. code = " << UINT8_TO_STRING(doipConfirm->result);
            doip_confirm_result_ = DoIPConfirmResult::CONFIRM_FAILED;
        }
        else {
            doip_confirm_result_ = DoIPConfirmResult::CONFIRM_OK;
        }
    }
}

void
DoIP_UdsTransportProtocolHandler::DoipIndicationCallback(doip_indication_t* doipIndication)
{
    DG_DEBUG << "DoIP_UdsTransportProtocolHandler::DoipIndicationCallback. ta_type " << doipIndication->ta_type
             << " result code: " << UINT8_TO_STRING(doipIndication->result)
             << " sa: " << UINT16_TO_STRING(doipIndication->logical_source_address)
             << " ta: " << UINT16_TO_STRING(doipIndication->logical_target_address);

    if ((nullptr == doipIndication) || (0 == doipIndication->data_length)) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipIndicationCallback doipIndication is nullptr or data length is 0.";
        return;
    }

    if (DOIP_RESULT::DOIP_RESULT_OK != doipIndication->result) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipIndicationCallback result error. code = " << UINT8_TO_STRING(doipIndication->result);
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (DiagServerTransPortService::getInstance()->GetDoipChannel() == Doip_Req_Channel::kNotSomeip)
    {
        DiagServerTransPortService::getInstance()->setDoipChannelEmpty();
    }
#endif
    // doip to update manager or remote
    if (DiagServerConfig::getInstance()->IsUpdateManagerOrRemoteAddress(doipIndication->logical_target_address)) {
        DiagServerRespUdsMessage udsmessage;
        udsmessage.udsSa = doipIndication->logical_source_address;
        udsmessage.udsTa = doipIndication->logical_target_address;
        udsmessage.busType = DiagUdsBusType::kDoip;
        udsmessage.result = 0;
        if (DOIP_RESULT::DOIP_RESULT_OK == doipIndication->result) {
            udsmessage.result = 1;
        }

        udsmessage.udsData.assign(doipIndication->data, doipIndication->data + doipIndication->data_length);
        DiagServerTransPortCM::getInstance()->DiagEventSend(udsmessage, DiagServerConfig::getInstance()->IsRemoteAddress(doipIndication->logical_target_address));
        return;
    }
#ifdef BUILD_SOMEIP_ENABLE
    // 如果目标地址是someip通道，且是来自someip通道的请求，则主动将消息发送回去，塞回someip通道
    if (DoSomeIPConfig::Instance()->IsDoSomeipProxyAddress(doipIndication->logical_target_address) &&  DiagServerTransPortService::getInstance()->GetDoipChannel() == Doip_Req_Channel::kSomeip) {
        DG_DEBUG << "doip ta is dosomeIp!";
        DoSomeIPRespUdsMessage udsmessage{};
        udsmessage.udsSa = doipIndication->logical_source_address;
        udsmessage.udsTa = doipIndication->logical_target_address;
        udsmessage.taType = static_cast<TargetAddressType>(doipIndication->ta_type);
        udsmessage.result = 0;
        if (DOIP_RESULT::DOIP_RESULT_OK == doipIndication->result) {
            udsmessage.result = 1;
        }
        udsmessage.udsData.assign(doipIndication->data, doipIndication->data + doipIndication->data_length);
        DiagServerTransPortService::getInstance()->ReplyUdsOnSomeIp(udsmessage, Req_Channel::kDoip);
        return;
    }
#endif

    UdsMessage::Address uds_sa = doipIndication->logical_source_address;
    UdsMessage::Address uds_ta = doipIndication->logical_target_address;
    UdsMessage::TargetAddressType request_ta = (UdsMessage::TargetAddressType)doipIndication->ta_type;
    ByteVector uds_payload;
    for (size_t i = 0; i < doipIndication->data_length; i++) {
        uds_payload.emplace_back(doipIndication->data[i]);
    }

    // for (auto& item : uds_payload) {
    //     DG_DEBUG << "DoIP_UdsTransportProtocolHandler::DoipIndicationCallback uds payload: " << UINT8_TO_STRING(item);
    // }

    ChannelID channel_id = 0;
    auto ret = transportProtocolManager_.IndicateMessage(
                                uds_sa, uds_ta, request_ta,
                                GlobalChannelIdentifier(GetHandlerId(), channel_id),
                                uds_payload.size(), 0, "DOIP",
                                uds_payload);

    if ((ret.first != UdsTransportProtocolMgr::IndicationResult::INDICATION_OK) || (ret.second == nullptr)) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipIndicationCallback indication result error. code = " << ret.first;
        // TODO reply transport specific error
        return;
    }

    uds_payload.clear();
    uds_payload = ret.second->GetUdsPayload();

    // TODO need check uds_payload data is complete or not
    bool bComplete = true;
    if (!bComplete) {
        transportProtocolManager_.NotifyMessageFailure(std::move(ret.second));
        // TODO reply transport specific error
        return;
    }

    transportProtocolManager_.HandleMessage(std::move(ret.second));
}

void
DoIP_UdsTransportProtocolHandler::DoipRouteCallback(doip_route_t* doipRoute)
{
    DG_DEBUG << "DoIP_UdsTransportProtocolHandler::DoipRouteCallback. ta_type " << doipRoute->ta_type
             << " result code: " << UINT8_TO_STRING(doipRoute->result)
             << " sa: " << UINT16_TO_STRING(doipRoute->logical_source_address)
             << " ta: " << UINT16_TO_STRING(doipRoute->logical_target_address);

    if ((nullptr == doipRoute) || (0 == doipRoute->data_length)) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipRouteCallback doipRoute is nullptr or data length is 0.";
        return;
    }

    if (DOIP_RESULT::DOIP_RESULT_OK != doipRoute->result) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipRouteCallback result error. code = " << UINT8_TO_STRING(doipRoute->result);
        return;
    }

    DiagServerReqUdsMessage udsmessage;
    udsmessage.udsSa = doipRoute->logical_source_address;
    udsmessage.udsTa = doipRoute->logical_target_address;
    udsmessage.busType = DiagUdsBusType::kDocan;
    udsmessage.taType = DiagTargetAddressType::kPhysical;
    for (size_t i = 0; i < doipRoute->data_length; i++) {
        udsmessage.udsData.emplace_back(doipRoute->data[i]);
    }

    bool bResult = DiagServerTransPortService::getInstance()->DoCanRequest(Docan_Instance, udsmessage, true, doipRoute->logical_source_address);
    if (!bResult) {
        DG_ERROR << "DoIP_UdsTransportProtocolHandler::DoipRouteCallback DoCanRequest failed.";
    }
}

}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
