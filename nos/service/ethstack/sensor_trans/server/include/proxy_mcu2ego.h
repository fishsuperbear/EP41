#pragma once

#include <memory>
#include "hozon/netaos/v1/mcudataservice_proxy.h"
#include "proto/fsm/function_manager.pb.h"

namespace hozon {
namespace netaos {
namespace sensor {
class Mcu2EgoProxy {
   public:
    Mcu2EgoProxy();
    ~Mcu2EgoProxy() = default;
    std::shared_ptr<hozon::functionmanager::FunctionManagerIn> Trans(
        ara::com::SamplePtr<::hozon::netaos::AlgMcuToEgoFrame const> data);
private:
    uint32_t _mcu2ego_pub_last_seq;
};

}  // namespace sensor
}  // namespace netaos
}  // namespace hozon