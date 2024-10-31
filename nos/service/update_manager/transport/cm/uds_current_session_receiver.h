#pragma once

#include "cm/include/proxy.h"
#include "idl/generated/diagPubSubTypes.h"
#include "update_manager/common/data_def.h"


namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

class UdCurrentSessionReceiver {
public:
    UdCurrentSessionReceiver();
    ~UdCurrentSessionReceiver();

    void Init();
    void DeInit();

    void EventCallback();

private:
    std::shared_ptr<Proxy> proxy_;
    std::shared_ptr<uds_current_session_notify_event> data_;
};


}  // namespace update
}  // namespace netaos
}  // namespace hozon