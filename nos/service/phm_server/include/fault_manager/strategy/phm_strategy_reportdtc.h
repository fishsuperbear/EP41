#pragma once

#include <mutex>
#include "proxy.h"
#include "skeleton.h"
#include "diagPubSubTypes.h"
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::cm;

class PhmStrategyReporterDtc : public StrategyBase
{
public:
    virtual void Init();
    virtual void DeInit();
    virtual void Act(const FaultInfo& faultData);

    PhmStrategyReporterDtc();
    virtual ~PhmStrategyReporterDtc();
    PhmStrategyReporterDtc(const PhmStrategyReporterDtc&) = delete;
    PhmStrategyReporterDtc& operator = (const PhmStrategyReporterDtc&) = delete;

private:
    std::shared_ptr<reportDemEventPubSubType> pubsubtype_;
    std::shared_ptr<Skeleton> skeleton_;
    std::shared_ptr<Proxy> proxy_;
    std::shared_ptr<reportDemEvent> dtc_data_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
