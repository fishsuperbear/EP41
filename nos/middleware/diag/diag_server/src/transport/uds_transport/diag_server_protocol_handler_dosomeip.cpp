/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server protocol handler(DoSomeIP)
*/

#include <functional>
#include <thread>

#include "diag/diag_server/include/transport/uds_transport/diag_server_protocol_handler_dosomeip.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/dosomeip/config/dosomeip_config.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace uds_transport {

using namespace hozon::netaos::diag::cm_transport;

DoSomeIP_UdsTransportProtocolHandler::DoSomeIP_UdsTransportProtocolHandler(const UdsTransportProtocolHandlerID handlerId,
        UdsTransportProtocolMgr& transportProtocolMgr)
: UdsTransportProtocolHandler(handlerId, transportProtocolMgr)
{
}

DoSomeIP_UdsTransportProtocolHandler::~DoSomeIP_UdsTransportProtocolHandler()
{
}

DoSomeIP_UdsTransportProtocolHandler::InitializationResult
DoSomeIP_UdsTransportProtocolHandler::Initialize() const
{
    DG_INFO << "DoSomeIP_UdsTransportProtocolHandler::Initialize.";
    return InitializationResult::INITIALIZE_OK;
}

bool
DoSomeIP_UdsTransportProtocolHandler::NotifyReestablishment() const
{
    return true;
}

bool
DoSomeIP_UdsTransportProtocolHandler::Start()
{
    DG_INFO << "DoSomeIP_UdsTransportProtocolHandler::Start.";
    bool bResult = DiagServerTransPortService::getInstance()->DoSomeIPStart(
                std::bind(&DoSomeIP_UdsTransportProtocolHandler::DoSomeIPUdsReqCallback, this, std::placeholders::_1),
                std::bind(&DoSomeIP_UdsTransportProtocolHandler::DoSomeIPLinkStatusCallback, this, std::placeholders::_1));
    return bResult;
}

void
DoSomeIP_UdsTransportProtocolHandler::Stop()
{
    DG_INFO << "DoSomeIP_UdsTransportProtocolHandler::Stop handlerId_: " << GetHandlerId() << ".";
    DiagServerTransPortService::getInstance()->DoSomeIPStop();
    transportProtocolManager_.HandlerStopped(GetHandlerId());
}

void
DoSomeIP_UdsTransportProtocolHandler::Transmit(UdsMessageConstPtr message, ChannelID transmitChannelId) const
{
    DG_DEBUG << "DoSomeIP_UdsTransportProtocolHandler::Transmit transmitChannelId: " << transmitChannelId << ".";
}

void
DoSomeIP_UdsTransportProtocolHandler::DoSomeIPUdsReqCallback(const DoSomeIPReqUdsMessage& req)
{
#ifdef BUILD_SOMEIP_ENABLE
    DG_DEBUG << "DoSomeIP_UdsTransportProtocolHandler::DoSomeIPUdsReqCallback. ta_type " << static_cast<uint16_t>(req.taType)
             << " sa: " << UINT16_TO_STRING(req.udsSa)
             << " ta: " << UINT16_TO_STRING(req.udsTa)
             << " uds data:" << UINT8_VEC_TO_STRING(req.udsData);

    DiagServerReqUdsMessage udsMessage{};
    udsMessage.udsSa = req.udsSa;
    udsMessage.udsTa = req.udsTa;
    udsMessage.taType = static_cast<DiagTargetAddressType>(req.taType);
    udsMessage.udsData.assign(req.udsData.begin(), req.udsData.end());


    if (DoSomeIPConfig::Instance()->IsDoIPAddress(req.udsTa))
    {
        DiagServerTransPortService::getInstance()->DoipRequestByEquip(udsMessage, true);
        return;
    }

    if (DoSomeIPConfig::Instance()->IsDoCanAddress(req.udsTa))
    {
        DiagServerTransPortService::getInstance()->DoCanRequest(Docan_Instance, udsMessage, false, 0x00, true);
        return;
    }
    DiagServerUdsMessage diagMsg{};
    diagMsg.udsSa = req.udsSa;
    diagMsg.udsTa = req.udsTa;
    diagMsg.taType = static_cast<DiagTargetAddressType>(req.taType);
    diagMsg.udsData.assign(req.udsData.begin(), req.udsData.end());

    DiagServerTransport::getInstance()->RecvUdsMessage(diagMsg, true);
#endif
}

void
DoSomeIP_UdsTransportProtocolHandler::DoSomeIPLinkStatusCallback(const DOSOMEIP_REGISTER_STATUS& status)
{
    DG_DEBUG << "DoSomeIP_UdsTransportProtocolHandler::DoSomeIPLinkStatusCallback, linkStatus is: " << static_cast<uint16_t>(status);
}

}  // namespace uds_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
