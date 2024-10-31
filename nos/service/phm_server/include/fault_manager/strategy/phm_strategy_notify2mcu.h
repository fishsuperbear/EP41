/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fault strategy notify to mcu
 */

#ifndef HZ_FM_STRATEGY_NOTIFY_MCU_H
#define HZ_FM_STRATEGY_NOTIFY_MCU_H

#include "phm_server/include/fault_manager/serviceInterface/phm_interface_fault2mcu.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {


class PhmStrategyNotify2Mcu : public StrategyBase
{
public:
    PhmStrategyNotify2Mcu();
    virtual ~PhmStrategyNotify2Mcu();

    void Init();
    void DeInit();

    void Act(const FaultInfo& faultData) override;
    void SendData(HzFaultEventToMCU& toMcuData);
    void UpdateTableItemFault(const int tableIndex, const uint32_t faultKey, const uint8_t faultStatus, int bitPosition);
    void UpdateClusterValue(const int tableIndex);

private:
    PhmStrategyNotify2Mcu(const PhmStrategyNotify2Mcu&);
    PhmStrategyNotify2Mcu& operator= (const PhmStrategyNotify2Mcu&);

    std::shared_ptr<PhmInterfaceFault2mcu> m_spPhmInterfaceFault2mcu;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // HZ_FM_STRATEGY_NOTIFY_MCU_H
