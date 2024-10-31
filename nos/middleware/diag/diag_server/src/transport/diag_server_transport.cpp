/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server transport
*/

#include <thread>

#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag_server_transport_cm.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerTransport* DiagServerTransport::instance_ = nullptr;
std::mutex DiagServerTransport::mtx_;

using namespace hozon::netaos::diag::cm_transport;

DiagServerTransport::DiagServerTransport()
: doip_transport_handler_(nullptr)
, doip_transport_mgr_(nullptr)
, docan_transport_handler_(nullptr)
, docan_transport_mgr_(nullptr)
, dosomeip_transport_handler_(nullptr)
, dosomeip_transport_mgr_(nullptr)
, stop_flag_(false)
,doserverReqChannel_(Server_Req_Channel::kDefault)
,time_mgr_(std::make_unique<TimerManager>())
{
    if (DiagServerConfig::getInstance()->IsSupportDoip()) {
        // doip Initialize
        uds_transport::UdsTransportProtocolHandlerID doip_handler_id = 0;
        doip_transport_mgr_ = new uds_transport::UdsTransportProtocolMgrImpl();
        doip_transport_handler_ = new uds_transport::DoIP_UdsTransportProtocolHandler(doip_handler_id, *doip_transport_mgr_);

        if (doip_transport_handler_ != nullptr) {
            doip_transport_handler_->Initialize();
        }
    }

    if (DiagServerConfig::getInstance()->IsSupportDoCan()) {
        // docan Initialize
        uds_transport::UdsTransportProtocolHandlerID docan_handler_id = 1;
        docan_transport_mgr_ = new uds_transport::UdsTransportProtocolMgrImpl();
        docan_transport_handler_ = new uds_transport::DoCAN_UdsTransportProtocolHandler(docan_handler_id, *docan_transport_mgr_);

        if (docan_transport_handler_ != nullptr) {
            docan_transport_handler_->Initialize();
        }
    }

    if (DiagServerConfig::getInstance()->IsSupportDoSomeip()) {
        // dosomeip Initialize
        uds_transport::UdsTransportProtocolHandlerID dosomeip_handler_id = 2;
        dosomeip_transport_mgr_ = new uds_transport::UdsTransportProtocolMgrImpl();
        dosomeip_transport_handler_ = new uds_transport::DoSomeIP_UdsTransportProtocolHandler(dosomeip_handler_id, *dosomeip_transport_mgr_);

        if (dosomeip_transport_handler_ != nullptr) {
            dosomeip_transport_handler_->Initialize();
        }
    }
}

DiagServerTransport*
DiagServerTransport::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerTransport();
        }
    }

    return instance_;
}

void
DiagServerTransport::Init()
{
    DG_INFO << "DiagServerTransport::Init";
    // doip start
    if (doip_transport_handler_ != nullptr) {
        bool ret = doip_transport_handler_->Start();
        if (!ret) {
            DG_ERROR << "DiagServerTransport::Init DoIP Transport Handler Start Failed.";
        }

        // TODO Post initialization failure processing
    }

    // docan start
    if (docan_transport_handler_ != nullptr) {
        bool ret = docan_transport_handler_->Start();
        if (!ret) {
            DG_ERROR << "DiagServerTransport::Init DoCAN Transport Handler Start Failed.";
        }

        // TODO Post initialization failure processing
    }

    // dosomeip start
    if (dosomeip_transport_handler_ != nullptr) {
        bool ret = dosomeip_transport_handler_->Start();
        if (!ret) {
            DG_ERROR << "DiagServerTransport::Init DoSomeIP Transport Handler Start Failed.";
        }

        // TODO Post initialization failure processing
    }

    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }
}

void
DiagServerTransport::DeInit()
{
    DG_INFO << "DiagServerTransport::DeInit";
    // doip stop
    if (doip_transport_handler_ != nullptr) {
        doip_transport_handler_->Stop();
    }

    // docan stop
    if (docan_transport_handler_ != nullptr) {
        docan_transport_handler_->Stop();
    }

    // docan stop
    if (dosomeip_transport_handler_ != nullptr) {
        dosomeip_transport_handler_->Stop();
    }

    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
        time_mgr_ = nullptr;
    }

    // need process transport_mgr_->HandlerStopped() after delete following point

    // TO DO(temporal strategy)
    {
        int iCount = 0;
        while(!stop_flag_) {
            if (iCount >= 10) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            iCount++;
        }
    }

    // release doip handler
    if (nullptr != doip_transport_handler_) {
        delete doip_transport_handler_;
        doip_transport_handler_ = nullptr;
    }

    if (nullptr != doip_transport_mgr_) {
        delete doip_transport_mgr_;
        doip_transport_mgr_ = nullptr;
    }

    // release docan handler
    if (nullptr != docan_transport_handler_) {
        delete docan_transport_handler_;
        docan_transport_handler_ = nullptr;
    }

    if (nullptr != docan_transport_mgr_) {
        delete docan_transport_mgr_;
        docan_transport_mgr_ = nullptr;
    }

    // release dosomeip handler
    if (nullptr != dosomeip_transport_handler_) {
        delete dosomeip_transport_handler_;
        dosomeip_transport_handler_ = nullptr;
    }

    if (nullptr != dosomeip_transport_mgr_) {
        delete dosomeip_transport_mgr_;
        dosomeip_transport_mgr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerTransport::SessionTimeout(void * data)
{
    DG_DEBUG << "DiagServerTransport::SessionTimeout switch isSomeipChannel_ To false";
    doserverReqChannel_ = Server_Req_Channel::kDefault;
}

void
DiagServerTransport::RecvUdsMessage(const DiagServerUdsMessage& udsMessage, const bool someipChannel)
{
    DG_DEBUG << "DiagServerTransport::RecvUdsMessage. udsData.size: " << udsMessage.udsData.size()
                                                                      <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                      <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    if (0 == udsMessage.udsData.size()) {
        DG_ERROR << "DiagServerTransport::RecvUdsMessage invalid udsMessage.udsData!";
        return;
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (doserverReqChannel_ != Server_Req_Channel::kDefault)
    {
        DG_ERROR << "Current RecvUdsMessage is busy ,ignore this req.";
        return;
    }

    if (someipChannel)
    {
        DG_DEBUG << "In SomeipChannel!";
        doserverReqChannel_ = Server_Req_Channel::kSomeip;
    } else {
        DG_DEBUG << "In NotSomeipChannel!";
        doserverReqChannel_ = Server_Req_Channel::kNotSomeip;
    }
    time_fd_ = -1;
    time_mgr_->StartFdTimer(time_fd_, 2000, std::bind(&DiagServerTransport::SessionTimeout, this, std::placeholders::_1), NULL, false);
#endif
    DiagServerUdsDataHandler::getInstance()->RecvUdsMessage(const_cast<DiagServerUdsMessage&>(udsMessage));
}

void
DiagServerTransport::ReplyUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransport::ReplyUdsMessage. udsData.size: " << udsMessage.udsData.size()
                                                                       <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                       <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    if (0 == udsMessage.udsData.size()) {
        DG_WARN << "DiagServerTransport::ReplyUdsMessage invalid udsMessage.udsData!";
        return;
    }

    // remote diag
    bool bResult = DiagServerConfig::getInstance()->IsRemoteAddress(udsMessage.udsTa);
    if (bResult) {
        DiagServerRespUdsMessage message;
        message.udsSa = udsMessage.udsSa;
        message.udsTa = udsMessage.udsTa;
        message.busType = DiagUdsBusType::kServer;
        message.result = 1;
        message.udsData.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());
        DiagServerTransPortCM::getInstance()->DiagEventSend(message, bResult);
        return;
    }
#ifdef BUILD_SOMEIP_ENABLE
    // 如果目标地址是someip通道，则主动将消息发送回去
    if (DoSomeIPConfig::Instance()->IsDoSomeipProxyAddress(udsMessage.udsTa) && doserverReqChannel_ == Server_Req_Channel::kSomeip) 
    {
        DG_DEBUG << "dosomeIp resp!";
        DoSomeIPRespUdsMessage udsmessage{};
        udsmessage.udsSa = udsMessage.udsSa;
        udsmessage.udsTa = udsMessage.udsTa;
        udsmessage.taType = static_cast<TargetAddressType>(udsMessage.taType);
        udsmessage.result = 0;
        udsmessage.udsData.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());
        DiagServerTransPortService::getInstance()->ReplyUdsOnSomeIp(udsmessage, Req_Channel::kServer);
        doserverReqChannel_ = Server_Req_Channel::kDefault;
        time_mgr_->StopFdTimer(time_fd_);
        time_fd_ = -1;
        return;
    }
#endif

    if (nullptr == doip_transport_handler_) {
        DG_ERROR << "DiagServerTransport::ReplyUdsMessage doip_transport_handler_ is nullptr. ";
        return;
    }
#ifdef BUILD_SOMEIP_ENABLE
    if (doserverReqChannel_ == Server_Req_Channel::kNotSomeip)
    {
        DG_DEBUG << "Server_Req_Channel kNotSomeip StopFdTimer";
        doserverReqChannel_ = Server_Req_Channel::kDefault;
        time_mgr_->StopFdTimer(time_fd_);
        time_fd_ = -1;
    }
#endif
    uds_transport::UdsMessage::Address udsSa = udsMessage.udsSa;
    uds_transport::UdsMessage::Address udsTa = udsMessage.udsTa;
    uds_transport::UdsMessage::TargetAddressType taType = static_cast<uds_transport::UdsMessage::TargetAddressType>(udsMessage.taType);
    uds_transport::ByteVector udsPayload;
    udsPayload.assign(udsMessage.udsData.begin(), udsMessage.udsData.end());
    const uds_transport::UdsMessage message = uds_transport::UdsMessageImpl(udsSa, udsTa, taType, udsPayload);
    uds_transport::UdsMessageConstPtr udsMessagePtr = std::make_unique<const uds_transport::UdsMessage>(message);
    DG_DEBUG << "DiagServerTransport::ReplyUdsMessage udsMessage. data_length: " << udsMessagePtr->GetUdsPayload().size()
                                                                                 <<  " sa: " << UINT16_TO_STRING(udsMessagePtr->GetSa())
                                                                                 <<  " ta: " << UINT16_TO_STRING(udsMessagePtr->GetTa());

    doip_transport_handler_->Transmit(std::move(udsMessagePtr), 0);
}

void
DiagServerTransport::HandlerStopped(const uint8_t handlerId)
{
    DG_DEBUG << "DiagServerTransport::HandlerStopped handlerId: " << handlerId << ".";
    // TO DO Processing of multiple handlers
    stop_flag_ = true;
}

void
DiagServerTransport::NotifyMessageFailure(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerTransport::NotifyMessageFailure. udsData.size: " << udsMessage.udsData.size()
                                                                            <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                            <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    // DiagServerUdsDataHandler::getInstance()->NotifyMessageFailure(udsMessage);
    // TODO: notify client failer
}

void
DiagServerTransport::TransmitConfirmation(const DiagServerUdsMessage& udsMessage, const bool confirmResult)
{
    DG_DEBUG << "DiagServerTransport::TransmitConfirmation. confirmResult: " << static_cast<uint>(confirmResult)
                                                                             << " udsData.size: " << udsMessage.udsData.size()
                                                                             <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                             <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    DiagServerUdsDataHandler::getInstance()->TransmitConfirmation(udsMessage, confirmResult);
    if (nullptr != doip_transport_handler_) {
        doip_transport_handler_->DoipResetConfirmResult();
    }
}

void
DiagServerTransport::NotifyDoipNetlinkStatus(const DoipNetlinkStatus doipNetlinkStatus, const uint16_t address)
{
    DG_DEBUG << "DiagServerTransport::NotifyDoipNetlinkStatus enter. doip netlink status " << doipNetlinkStatus
                                                                    << " client address " << address;

    DiagServerSessionHandler::getInstance()->OnDoipNetlinkStatusChange(doipNetlinkStatus, address);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
