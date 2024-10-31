#include <cstddef>
#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_notify2mcu.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_notify2app.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_collect_res.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_restartproc.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_reportdtc.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_mgr.h"

namespace hozon {
namespace netaos {
namespace phm_server {

PhmStrategyMgr* PhmStrategyMgr::instance_ = nullptr;
std::mutex PhmStrategyMgr::mtx_;

PhmStrategyMgr*
PhmStrategyMgr::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PhmStrategyMgr();
        }
    }

    return instance_;
}

PhmStrategyMgr::PhmStrategyMgr()
: m_spRestartProc(new PhmStrategyRestartProc())
, m_spCollectRes(nullptr)
, m_spReporterDtc(new PhmStrategyReporterDtc())
, m_spNotifyMcu(new PhmStrategyNotify2Mcu())
, m_spNotifyApp(new PhmStrategyNotify2App())
{
}

void
PhmStrategyMgr::Init()
{
    PHMS_INFO << "PhmStrategyMgr::Init";
    FaultTaskHandler::getInstance()->RegisterStrategyCallback(std::bind(&PhmStrategyMgr::StrategyFaultCallback, this, std::placeholders::_1));
    if (m_spRestartProc) {
        m_spRestartProc->Init();
    }
    if (m_spCollectRes) {
        m_spCollectRes->Init();
    }
    if (m_spReporterDtc) {
        m_spReporterDtc->Init();
    }
    if (m_spNotifyMcu) {
        m_spNotifyMcu->Init();
    }
    if (m_spNotifyApp) {
        m_spNotifyApp->Init();
    }
}

void
PhmStrategyMgr::DeInit()
{
    PHMS_INFO << "PhmStrategyMgr::DeInit";
    if (m_spNotifyApp != nullptr) {
        m_spNotifyApp->DeInit();
    }

    if (m_spNotifyMcu != nullptr) {
        m_spNotifyMcu->DeInit();
    }

    if (m_spReporterDtc != nullptr) {
        m_spReporterDtc->DeInit();
    }

    if (m_spCollectRes != nullptr) {
        m_spCollectRes->DeInit();
    }

    if (m_spRestartProc != nullptr) {
        m_spRestartProc->DeInit();
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
PhmStrategyMgr::StrategyFaultCallback(Fault_t fault)
{
    uint32_t faultKey = fault.faultId*100 + fault.faultObj;
    PHMS_INFO << "PhmStrategyMgr::StrategyFaultCallback fault " << faultKey;
    FaultInfo faultInfo;
    if (!PHMServerConfig::getInstance()->GetFaultInfoByFault(faultKey, faultInfo)) {
        PHMS_ERROR << "PhmStrategyMgr::StrategyFaultCallback not support fault: " << faultKey;
        return;
    }

    if (1 == faultInfo.faultAction.strategy.notifyMcu) {
        if (m_spNotifyMcu) m_spNotifyMcu->Act(faultInfo);
    }

    if (1 == faultInfo.faultAction.strategy.notifyApp) {
        if (m_spNotifyApp) m_spNotifyApp->Act(faultInfo);
    }

    if (1 == faultInfo.faultAction.strategy.restartproc) {
        if (m_spRestartProc) m_spRestartProc->Act(faultInfo);
    }

    if (1 == faultInfo.faultAction.strategy.dtcMapping) {
        if (m_spReporterDtc) m_spReporterDtc->Act(faultInfo);
    }

    if (m_spCollectRes) {
        m_spCollectRes->Act(faultInfo);
    }
    PHMS_INFO << "PhmStrategyMgr::StrategyFaultCallback fault " << faultKey << " finish!";
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
