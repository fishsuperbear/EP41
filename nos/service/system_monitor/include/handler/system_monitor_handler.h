/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor handler
 */

#ifndef SYSTEM_MONITOR_HANDLER_H
#define SYSTEM_MONITOR_HANDLER_H

#include <mutex>
#include "phm/include/phm_client.h"
#include "system_monitor/include/transport/system_monitor_transport_event_receiver.h"
#include "system_monitor/include/common/system_monitor_def.h"

using namespace hozon::netaos::phm;

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorHandler {
public:
    static SystemMonitorHandler* getInstance();

    void Init();
    void DeInit();

    void ControlEventCallBack(const SystemMonitorControlEventInfo& info);
    void RefreshEventCallback(const std::string& reason);

    bool ReportFault(const SystemMonitorSendFaultInfo& faultInfo);

private:
    static std::string GetVinNumber();
    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);

private:
    SystemMonitorHandler();
    SystemMonitorHandler(const SystemMonitorHandler &);
    SystemMonitorHandler & operator = (const SystemMonitorHandler &);

private:
    static std::mutex mtx_;
    static SystemMonitorHandler* instance_;

    PHMClient* phm_client_;

    SystemMonitorTransportEventReceiver* event_receiver_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_HANDLER_H