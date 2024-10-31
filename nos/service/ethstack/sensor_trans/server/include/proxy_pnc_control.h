#pragma once
#include <memory>
#include "proto/statemachine/state_machine.pb.h"
#include "hozon/netaos/v1/mcudataservice_proxy.h"

namespace hozon {
namespace netaos {
namespace sensor {
class PncCtrProxy {
public:
    PncCtrProxy();
    ~PncCtrProxy() = default;
    std::shared_ptr<hozon::state::StateMachine> Trans(
        ara::com::SamplePtr<::hozon::netaos::PNCControlState const> data);
private:
    uint32_t _pnc_ctr_seqid;
};



}   // namespace sensor
}   // namespace netaos
}   // namespace hozon