#ifndef SYSTEM_MONITOR_TRANSPORT_EVENT_RECEIVER_H
#define SYSTEM_MONITOR_TRANSPORT_EVENT_RECEIVER_H

#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

using namespace hozon::netaos::cm;

class SystemMonitorTransportEventReceiver {
public:
    SystemMonitorTransportEventReceiver();
    ~SystemMonitorTransportEventReceiver();

    void Init(const std::string& vin);
    void DeInit();

private:
    void ControlEventCallback();
    void RefreshEventCallback();

private:
    std::shared_ptr<Proxy> control_proxy_;
    std::shared_ptr<Proxy> refresh_proxy_;

};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_TRANSPORT_EVENT_RECEIVER_H