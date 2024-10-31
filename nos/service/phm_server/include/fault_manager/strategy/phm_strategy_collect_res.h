/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: util
 */

#pragma once
#include "phm_server/include/common/phm_server_def.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_base.h"
#include <string>
#include <mutex>
#include <memory>

namespace hozon {
namespace netaos {
namespace phm_server {

struct FmCollectData
{
    uint32_t faultId;
    uint32_t faultObj;
    std::string collectLevel;
    uint8_t isFaultTrigger;
};


class ThreadPool;
class TimerManager;
class PhmStrategyCollectRes : public StrategyBase
{
public:
    PhmStrategyCollectRes();
    ~PhmStrategyCollectRes();

    virtual void Init();
    virtual void DeInit();
    virtual void Act(const FaultInfo& faultData);

    void CheckFaultClusterLevel(const uint32_t fault_key);
    void SetResCollectLevel(uint8_t level);
    void StartResourceScriptExt();
    void ResourceTimeout(void * data);
    void StartResourceScript(const FmCollectData* pFmCollectData);
    void DoResourceCollectTask(const FmCollectData* pcFmCollectData);

private:
    PhmStrategyCollectRes(const PhmStrategyCollectRes&);
    PhmStrategyCollectRes& operator=(const PhmStrategyCollectRes&);

    std::mutex mtx_;
    bool start_collect_flag_;
    int res_col_fd_;
    uint32_t res_collect_level_;

    std::shared_ptr<ThreadPool> m_spThreadPool;
    std::shared_ptr<TimerManager> m_spTimerManager;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

