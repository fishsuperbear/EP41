
#ifndef PHM_STRATEGY_MGR_H
#define PHM_STRATEGY_MGR_H

#include <mutex>
#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class PhmStrategyMgr {

public:
    static PhmStrategyMgr* getInstance();

    void Init();
    void DeInit();

private:
    PhmStrategyMgr();
    PhmStrategyMgr(const PhmStrategyMgr &);
    PhmStrategyMgr & operator = (const PhmStrategyMgr &);

    void StrategyFaultCallback(Fault_t fault);

private:
    static std::mutex mtx_;
    static PhmStrategyMgr* instance_;

    std::shared_ptr<StrategyBase> m_spRestartProc;
    std::shared_ptr<StrategyBase> m_spCollectRes;
    std::shared_ptr<StrategyBase> m_spReporterDtc;
    std::shared_ptr<StrategyBase> m_spNotifyMcu;
    std::shared_ptr<StrategyBase> m_spNotifyApp;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_STRATEGY_MGR_H
