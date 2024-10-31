 #include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_reportdtc.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"

namespace hozon {
namespace netaos {
namespace phm_server {


PhmStrategyReporterDtc::PhmStrategyReporterDtc()
{
}

PhmStrategyReporterDtc::~PhmStrategyReporterDtc()
{
}

void
PhmStrategyReporterDtc::Init()
{
    PHMS_INFO << "PhmStrategyReporterDtc::Init";
    pubsubtype_ = std::make_shared<reportDemEventPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(0, "reportDemEvent");
}

void
PhmStrategyReporterDtc::DeInit()
{
    PHMS_INFO << "PhmStrategyReporterDtc::DeInit";
}

void
PhmStrategyReporterDtc::Act(const FaultInfo& faultData)
{
    uint32_t faultKey = faultData.faultId*100 + faultData.faultObj;
    PHMS_INFO << "PhmStrategyReporterDtc::Act fault " << faultKey
              << " dtcCode: " << UINT32_TO_STRING(faultData.dtcCode);
    if (!faultData.dtcCode) {
        return;
    }

    std::shared_ptr<reportDemEvent> data = std::make_shared<reportDemEvent>();
    data->dtcValue(faultKey);
    data->alarmStatus(faultData.faultStatus);
    if (skeleton_->IsMatched()) {
        if (skeleton_->Write(data) != 0) {
            PHMS_WARN << "PhmStrategyReporterDtc::Act send data failed!";
        }
    }
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
