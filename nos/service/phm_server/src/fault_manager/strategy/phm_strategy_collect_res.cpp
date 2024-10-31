/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: util
*/
#include "phm_server/include/common/thread_pool.h"
#include "phm_server/include/common/time_manager.h"
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_collect_res.h"


namespace hozon {
namespace netaos {
namespace phm_server {

const std::string RES_COLLECT_TASK = "PhmStrategyCollectRes";
const unsigned int RES_COLLECT_TIME = 20000; // 20s


PhmStrategyCollectRes::PhmStrategyCollectRes()
: start_collect_flag_(false)
, res_col_fd_(-1)
, res_collect_level_(3)
, m_spThreadPool(new ThreadPool(1))
, m_spTimerManager(new TimerManager())
{
    m_spTimerManager->Init();
}

PhmStrategyCollectRes::~PhmStrategyCollectRes()
{
    PHMS_INFO << "PhmStrategyCollectRes::~PhmStrategyCollectRes";
    if (m_spTimerManager) {
        m_spTimerManager->DeInit();
        m_spTimerManager = nullptr;
    }

    if (m_spThreadPool) {
        m_spThreadPool->Stop();
        m_spThreadPool = nullptr;
    }
}

void
PhmStrategyCollectRes::Init()
{
    PHMS_INFO << "PhmStrategyCollectRes::Init";
}

void
PhmStrategyCollectRes::DeInit()
{
    PHMS_INFO << "PhmStrategyCollectRes::DeInit fd:" << res_col_fd_;
    m_spTimerManager->StopFdTimer(res_col_fd_);
}

void
PhmStrategyCollectRes::Act(const FaultInfo& faultData)
{
    uint32_t faultKey = faultData.faultId * 100 + faultData.faultObj;
    CheckFaultClusterLevel(faultKey);
    return;
}

void
PhmStrategyCollectRes::CheckFaultClusterLevel(const uint32_t faultKey)
{
    PHMS_INFO << "PhmStrategyCollectRes::CheckFaultClusterLevel faultKey:" << faultKey;
    std::vector<FaultClusterItem> vFaultCluster;
    bool isHasCluster = PHMServerConfig::getInstance()->getFaultCluster(faultKey, vFaultCluster);
    if (!isHasCluster) {
        // PHMS_DEBUG << "PhmStrategyCollectRes::CheckFaultClusterLevel key not have cluster";
        return;
    }

    std::lock_guard<std::mutex> lck(mtx_);
    if (start_collect_flag_) {
        PHMS_INFO << "PhmStrategyCollectRes::CheckFaultClusterLevel is running!";
        return;
    }

    // TODO
    // for (auto& item : vFaultCluster) {
    //     if (item.bitPosition >= res_collect_level_) {
    //         PHMS_INFO << "fm start collect event by more than " << res_collect_level_ << " level, faultKey " << faultKey << "and start resource script";
    //         if (m_spTimerManager) {
    //             m_spTimerManager->StartFdTimer(res_col_fd_, RES_COLLECT_TIME,
    //                 std::bind(&PhmStrategyCollectRes::ResourceTimeout, this, std::placeholders::_1), nullptr);
    //         }

    //         FmCollectData* pcFmCollectData = new FmCollectData();
    //         pcFmCollectData->faultId = faultKey / 100;
    //         pcFmCollectData->faultObj = faultKey % 100;
    //         pcFmCollectData->collectLevel = std::to_string(item.bitPosition);
    //         pcFmCollectData->isFaultTrigger = 0x01;
    //         StartResourceScript(pcFmCollectData);
    //         break;
    //     }
    // }
}

void
PhmStrategyCollectRes::SetResCollectLevel(uint8_t level)
{
    if (level > 8) {
        PHMS_WARN << "PhmStrategyCollectRes::SetResCollectLevel failed! input level > 8";
        return;
    }

    res_collect_level_ = level;
    PHMS_INFO << "PhmStrategyCollectRes::SetResCollectLevel " << res_collect_level_ << " success!";
}

void
PhmStrategyCollectRes::StartResourceScriptExt()
{
    std::lock_guard<std::mutex> lck(mtx_);
    if (start_collect_flag_) {
        PHMS_INFO << "PhmStrategyCollectRes::ResourceScript is running!";
        return;
    }

    PHMS_INFO << "PhmStrategyCollectRes::ResourceScript execute by data collect!";
    if (m_spTimerManager) {
        m_spTimerManager->StartFdTimer(res_col_fd_, RES_COLLECT_TIME,
            std::bind(&PhmStrategyCollectRes::ResourceTimeout, this, std::placeholders::_1), nullptr);
    }

    FmCollectData* pcFmCollectData = new FmCollectData();
    pcFmCollectData->faultId = 0x00;
    pcFmCollectData->faultObj = 0x00;
    pcFmCollectData->collectLevel = std::to_string(res_collect_level_);
    pcFmCollectData->isFaultTrigger = 0x00;
    StartResourceScript(pcFmCollectData);
    return;
}

void
PhmStrategyCollectRes::ResourceTimeout(void * data)
{
    PHMS_INFO << "PhmStrategyCollectRes::ResourceScript end!";
    start_collect_flag_ = false;
    res_col_fd_ = -1;
}

void
PhmStrategyCollectRes::StartResourceScript(const FmCollectData* pFmCollectData)
{
    PHMS_INFO << "PhmStrategyCollectRes::ResourceScript start!";
    start_collect_flag_ = true;
    if (m_spThreadPool) m_spThreadPool->Commit(std::bind(&PhmStrategyCollectRes::DoResourceCollectTask, this, pFmCollectData));
    return;
}

void
PhmStrategyCollectRes::DoResourceCollectTask(const FmCollectData* pcFmCollectData)
{
    PHMS_DEBUG << "PhmStrategyCollectRes::DoResourceCollectTask Run ";
    const std::string cmd = "bash /app/script/res_statistic.sh "
                            + std::string(" fault_monitor ")
                            + std::to_string(pcFmCollectData->faultId) + " "
                            + std::to_string(pcFmCollectData->faultObj) + " "
                            + pcFmCollectData->collectLevel + " "
                            + std::to_string(pcFmCollectData->isFaultTrigger) + " & ";
    PHMS_INFO << "PhmStrategyCollectRes::DoResourceCollectTask Run cmd:" << cmd;
    delete pcFmCollectData;
    pid_t status = system(cmd.data());
    if (status == -1) {
        PHMS_ERROR << "PhmStrategyCollectRes::DoResourceCollectTask Run res_statistic: system error!";
    }

    return;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
