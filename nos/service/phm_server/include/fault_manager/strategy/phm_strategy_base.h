/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fault strategy base
 */

#ifndef PHM_STRATEGY_BASE_H
#define PHM_STRATEGY_BASE_H

#include "phm_server/include/common/phm_server_def.h"
#include <stdint.h>

namespace hozon {
namespace netaos {
namespace phm_server {

class StrategyBase {
public:
    StrategyBase() {};
    virtual ~StrategyBase() {};
    virtual void Act(const FaultInfo& faultData) = 0;
    virtual void Init() {}
    virtual void DeInit() {}
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_STRATEGY_BASE_H
