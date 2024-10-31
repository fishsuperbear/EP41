#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/strategy/phm_strategy_mgr.h"
#include "phm_server/include/fault_manager/manager/phm_fault_manager.h"
#include "phm_server/include/fault_manager/manager/phm_fault_dispatcher.h"
#include "phm_server/include/fault_manager/manager/phm_fault_record.h"
#include "phm_server/include/fault_manager/manager/phm_fault_task_handler.h"
#include "phm_server/include/fault_manager/manager/phm_config_monitor.h"
#include "phm_server/include/fault_manager/manager/phm_collect_file.h"
#include "phm_server/include/fault_manager/interactive/phm_fault_diag_handler.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"
#include "phm_server/include/fault_manager/analysis/fault_analysis.h"
#ifdef BUILD_FOR_ORIN
    #include <ara/core/initialization.h>
    #include "phm_server/include/fault_manager/serviceInterface/phm_interface_faultfrommcu.h"
#endif

namespace hozon {
namespace netaos {
namespace phm_server {

FaultManager* FaultManager::instance_ = nullptr;
std::mutex FaultManager::mtx_;

FaultManager*
FaultManager::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new FaultManager();
        }
    }

    return instance_;
}

FaultManager::FaultManager()
{
    m_spPhmCollectFile = std::make_shared<PhmCollectFile>();
}

void
FaultManager::Init()
{
    PHMS_INFO << "FaultManager::Init enter!";
#ifdef BUILD_FOR_ORIN
    ara::core::Initialize();
    PhmInterfaceFaultFromMcu::getInstance()->OfferService();
#endif

    FileOperate::getInstance()->Init();
    FaultTaskHandler::getInstance()->Init();
    PhmCfgMonitor::getInstance()->Init();
    FaultDispatcher::getInstance()->Init();
    FaultAnalysis::getInstance()->Init();
    FaultRecorder::getInstance()->Init();
    PhmStrategyMgr::getInstance()->Init();
    DiagMessageHandler::getInstance()->Init();
    m_spPhmCollectFile->Init();
    PHMS_INFO << "FaultManager::Init done!";
}

void
FaultManager::DeInit()
{
    PHMS_INFO << "FaultManager::DeInit enter!";
    m_spPhmCollectFile->DeInit();
    DiagMessageHandler::getInstance()->DeInit();
    PhmStrategyMgr::getInstance()->DeInit();
    FaultRecorder::getInstance()->DeInit();
    FaultAnalysis::getInstance()->DeInit();
    FaultDispatcher::getInstance()->DeInit();
    PhmCfgMonitor::getInstance()->DeInit();
    FaultTaskHandler::getInstance()->DeInit();
    FileOperate::getInstance()->DeInit();

#ifdef BUILD_FOR_ORIN
    PhmInterfaceFaultFromMcu::getInstance()->StopOfferService();
    ara::core::Deinitialize();
#endif

    PHMS_INFO << "FaultManager::DeInit done!";
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
