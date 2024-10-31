/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fault strategy notify to app
 */

#ifndef HZ_FM_STRATEGY_NOTIFY_APP_H
#define HZ_FM_STRATEGY_NOTIFY_APP_H

#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {


class PhmStrategyNotify2App : public StrategyBase
{
public:
    PhmStrategyNotify2App();
    virtual ~PhmStrategyNotify2App();

    void Init();
    void DeInit();
    void Act(const FaultInfo& faultData) override;

private:
    PhmStrategyNotify2App(const PhmStrategyNotify2App&);
    PhmStrategyNotify2App& operator= (const PhmStrategyNotify2App&);
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // HZ_FM_STRATEGY_NOTIFY_APP_H
