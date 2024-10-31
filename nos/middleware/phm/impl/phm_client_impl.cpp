#include "phm/impl/include/phm_client_impl.h"
#include "phm/common/include/phm_logger.h"
#include "phm/common/include/phm_config.h"
#include "phm/common/include/timer_manager.h"
#include "phm/fault_manager/include/fault_reporter.h"


namespace hozon {
namespace netaos {
namespace phm {

PHMClientImpl::PHMClientImpl()
: is_stoped_(false)
{
    PHM_DEBUG << "PHMClientImpl::PHMClientImpl";
    m_isInit = false;
    PHMConfig::getInstance()->LoadPhmConfig();
    TimerManager::Instance();

    m_spFaultReporter = std::make_shared<FaultReporter>();
    module_cfg_ = std::make_shared<ModuleConfig>();
    task_manager_ = std::make_unique<PHMTaskManager>(module_cfg_, m_spFaultReporter);
    m_spFaultCollectionFile = std::make_shared<FaultCollectionFile>();
}

PHMClientImpl::~PHMClientImpl()
{
    PHM_DEBUG << "PHMClientImpl::~PHMClientImpl -";
    if (!is_stoped_) {
        this->Stop();
    }

    if (m_spFaultCollectionFile) {
        m_spFaultCollectionFile->Deinit();
        m_spFaultCollectionFile = nullptr;
    }

    this->Deinit();

    if (thread_init_.joinable()) {
        thread_init_.join();
    }

    PHM_DEBUG << "PHMClientImpl::~PHMClientImpl +";
}

int32_t
PHMClientImpl::Init(const std::string& phmConfigPath,
                    std::function<void(bool)> service_available_callback,
                    std::function<void(ReceiveFault_t)> fault_receive_callback,
                    const std::string& processName)
{
    PHM_DEBUG << "PHMClientImpl::Init";
    if (true == m_isInit.load()) {
        PHM_DEBUG << "PHMClientImpl::Init already";
        return -1;
    }

    int32_t res1 = PHMConfig::getInstance()->Init(phmConfigPath, module_cfg_);
    int32_t res2 = TimerManager::Instance()->Init();
    int32_t res3 = m_spFaultReporter->Init(fault_receive_callback, processName);
    int32_t ret = ((0 == res1) & (0 == res2) & (0 == res3)) ? 0 : -1;
    PHM_DEBUG << "PHMClientImpl::Init ret" << ret;
    FaultReceiveTable::getInstance()->Set(module_cfg_, fault_receive_callback);
    m_spFaultCollectionFile->Init();

    if (nullptr != service_available_callback) {
        thread_init_ = std::thread([=]() {
            (ret == 0) ? service_available_callback(true)
                       : service_available_callback(false);
        });
       thread_init_.detach();
    }

    m_isInit.store(true);
    PHM_DEBUG << "PHMClientImpl::Init finish!";
    return ret;
}

int32_t
PHMClientImpl::Start(uint32_t delayTime)
{
    if (false == m_isInit.load()) {
        PHM_INFO << "PHMClientImpl::Start need init first";
        return false;
    }

    if (nullptr == task_manager_) {
        PHM_ERROR << "PHMClientImpl::Start task_manager_ is nullptr!";
        return -1;
    }

    int32_t res = task_manager_->StartAllTask(module_cfg_->GetPhmTask(), delayTime);
    return res;
}

void
PHMClientImpl::Stop()
{
    PHM_DEBUG << "PHMClientImpl::Stop";
    is_stoped_ = true;

    if (task_manager_!= nullptr) {
        task_manager_->StopAllTask();
    }
}

void
PHMClientImpl::Deinit()
{
    if (false == m_isInit.load()) {
        PHM_INFO << "PHMClientImpl::DeInit already finished";
        return;
    }
    m_isInit.store(false);

    TimerManager::Instance()->DeInit();
    m_spFaultReporter->DeInit();
    PHM_DEBUG << "PHMClientImpl::Deinit finish!";
}

int32_t
PHMClientImpl::ReportCheckPoint(const uint32_t checkPointId)
{
    if (false == m_isInit.load()) {
        PHM_INFO << "PHMClientImpl::ReportCheckPoint need init first";
        return false;
    }

    if (nullptr == task_manager_) {
        PHM_ERROR << "PHMClientImpl::ReportCheckPoint task_manager_ is nullptr!";
        return -1;
    }

    return task_manager_->ReportCheckPoint(checkPointId);
}

int32_t
PHMClientImpl::ReportFault(const SendFault_t& faultInfo)
{
    if (false == m_isInit.load()) {
        PHM_INFO << "PHMClientImpl::ReportFault need init first";
        return false;
    }

    m_spFaultReporter->ReportFault(faultInfo, module_cfg_);
    return 0;
}

void
PHMClientImpl::InhibitFault(const std::vector<uint32_t>& faultKeys)
{
    module_cfg_->InhibitFault(faultKeys);
}

void
PHMClientImpl::RecoverInhibitFault(const std::vector<uint32_t>& faultKeys)
{
    module_cfg_->RecoverInhibitFault(faultKeys);
}

void
PHMClientImpl::InhibitAllFault()
{
    module_cfg_->InhibitAllFault();
}

void
PHMClientImpl::RecoverInhibitAllFault()
{
    module_cfg_->RecoverInhibitAllFault();
}

int32_t
PHMClientImpl::GetDataCollectionFile(std::function<void(std::vector<std::string>&)> collectionFileCb)
{
    if (false == m_isInit.load()) {
        PHM_INFO << "PHMClientImpl::GetDataCollectionFile need init first";
        return false;
    }

    m_spFaultCollectionFile->Request(RequestTypeGetCollectFile, collectionFileCb);
    return 0;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
/* EOF */
