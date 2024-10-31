/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: tpl event receiver
*/

#include "idl/generated/monitorPubSubTypes.h"
#include "system_monitor/include/transport/system_monitor_transport_event_receiver.h"
#include "system_monitor/include/handler/system_monitor_handler.h"
#include "system_monitor/include/common/system_monitor_logger.h"

namespace hozon {
namespace netaos {
namespace system_monitor {


SystemMonitorTransportEventReceiver::SystemMonitorTransportEventReceiver()
: control_proxy_(nullptr)
, refresh_proxy_(nullptr)
{
}

SystemMonitorTransportEventReceiver::~SystemMonitorTransportEventReceiver()
{
}

void
SystemMonitorTransportEventReceiver::Init(const std::string& vin)
{
    STMM_INFO << "SystemMonitorTransportEventReceiver::Init vin: " << vin;
    // control proxy init
    std::shared_ptr<monitor_control_eventPubSubType> controlPubsubtype = std::make_shared<monitor_control_eventPubSubType>();
    control_proxy_ = std::make_shared<Proxy>(controlPubsubtype);
    control_proxy_->Init(0, "monitor_control_eventTopic_" + vin);
    control_proxy_->Listen(std::bind(&SystemMonitorTransportEventReceiver::ControlEventCallback, this));

    // refresh proxy init
    std::shared_ptr<monitor_refresh_eventPubSubType> refreshPubsubtype = std::make_shared<monitor_refresh_eventPubSubType>();
    refresh_proxy_ = std::make_shared<Proxy>(refreshPubsubtype);
    refresh_proxy_->Init(0, "monitor_refresh_event");
    refresh_proxy_->Listen(std::bind(&SystemMonitorTransportEventReceiver::RefreshEventCallback, this));
}

void
SystemMonitorTransportEventReceiver::DeInit()
{
    STMM_INFO << "SystemMonitorTransportEventReceiver::DeInit";
    // refresh proxy deinit
    if (nullptr != refresh_proxy_) {
        refresh_proxy_->Deinit();
        refresh_proxy_ = nullptr;
    }

    // control proxy deinit
    if (nullptr != control_proxy_) {
        control_proxy_->Deinit();
        control_proxy_ = nullptr;
    }
}

void
SystemMonitorTransportEventReceiver::ControlEventCallback()
{
    STMM_DEBUG << "SystemMonitorTransportEventReceiver::ControlEventCallback.";
    if (nullptr == control_proxy_) {
        STMM_ERROR << "SystemMonitorTransportEventReceiver::ControlEventCallback control_proxy_ is nullptr.";
        return;
    }

    if (control_proxy_->IsMatched()) {
        std::shared_ptr<monitor_control_event> data = std::make_shared<monitor_control_event>();
        control_proxy_->Take(data);
        STMM_INFO << "SystemMonitorTransportEventReceiver::ControlEventCallback monitor_id: " << data->monitor_id()
                                                                         << " control_type: " << data->control_type()
                                                                         << " control_value: " << data->control_value();
        SystemMonitorControlEventInfo eventInfo;
        eventInfo.id = static_cast<SystemMonitorSubFunctionId>(data->monitor_id());
        eventInfo.type = static_cast<SystemMonitorSubFunctionControlType>(data->control_type());
        eventInfo.value = data->control_value();
        SystemMonitorHandler::getInstance()->ControlEventCallBack(eventInfo);
    }
}

void
SystemMonitorTransportEventReceiver::RefreshEventCallback()
{
    STMM_DEBUG << "SystemMonitorTransportEventReceiver::RefreshEventCallback.";
    if (nullptr == refresh_proxy_) {
        STMM_ERROR << "SystemMonitorTransportEventReceiver::RefreshEventCallback refresh_proxy_ is nullptr.";
        return;
    }

    if (refresh_proxy_->IsMatched()) {
        std::shared_ptr<monitor_refresh_event> data = std::make_shared<monitor_refresh_event>();
        refresh_proxy_->Take(data);
        STMM_INFO << "SystemMonitorTransportEventReceiver::RefreshEventCallback refresh_reason: " << data->refresh_reason();
        SystemMonitorHandler::getInstance()->RefreshEventCallback(std::string(data->refresh_reason()));
    }
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon