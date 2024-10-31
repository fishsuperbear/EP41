#include "update_manager/agent/ota_agent.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

OTAAgent* OTAAgent::m_pInstance = nullptr;
std::mutex OTAAgent::m_mtx;


OTAAgent::OTAAgent()
    :update_interface_req_dispatcher_(nullptr)
{
    update_interface_req_dispatcher_ = std::make_unique<InterfaceUpdateReqDispatcher>();
}

OTAAgent::~OTAAgent()
{
}

OTAAgent*
OTAAgent::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new OTAAgent();
        }
    }

    return m_pInstance;
}

void
OTAAgent::Init()
{
    UM_INFO << "OTAAgent::Init.";
    update_interface_req_dispatcher_->Init();
    UM_INFO << "OTAAgent::Init Done.";
}

void
OTAAgent::Deinit()
{
    UM_INFO << "OTAAgent::Deinit.";
    update_interface_req_dispatcher_->Deinit();

    update_interface_req_dispatcher_ = nullptr;

    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "OTAAgent::Deinit Done.";
}

bool OTAAgent::Update(const std::string& packageName)
{
    if (update_interface_req_dispatcher_ == nullptr) {
        return false;
    }
    auto res = update_interface_req_dispatcher_->Update(packageName);
    if (res > 0) {
        UPDATE_LOG_E("update failed, error code is : %d", res);
    }
    return true;
}

bool OTAAgent::GetVersionInfo(std::string& mdcVersion)
{
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->GetVersionInfo(mdcVersion) != 0) {
        return false;
    }
    return true;
}


bool OTAAgent::Activate()
{
#ifdef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->Activate() != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::Query(std::string& updateStatus)
{
#ifdef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->Query(updateStatus) != 0) {
        return false;
    }
#else
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->QueryUpdateStatus(updateStatus) != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::Finish()
{
#ifdef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->Finish() != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::GetActivationProgress(uint8_t& progress, std::string& message)
{
#ifdef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->GetActivationProgress(progress, message) != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::GetUpdateProgress(uint8_t& progress, std::string& message)
{
#ifdef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->GetUpdateProgress(progress, message) != 0) {
        return false;
    }
#else
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->QueryUpdateProgress(progress) != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::QueryStatus(std::string& updateStatus, uint8_t& progress)
{
#ifndef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->Query(updateStatus, progress) != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::SwitchSlot()
{
#ifndef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->SwitchSlot() != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::GetCurrentSlot(std::string& currentSlot)
{
#ifndef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->GetCurrentSlot(currentSlot) != 0) {
        return false;
    }
#endif
    UM_DEBUG << "current partition is : " << currentSlot;
    return true;
}

bool OTAAgent::Reboot()
{
#ifndef BUILD_FOR_MDC
    if (update_interface_req_dispatcher_ == nullptr
        || update_interface_req_dispatcher_->Reboot() != 0) {
        return false;
    }
#endif
    return true;
}

bool OTAAgent::HardReboot()
{
    UPDATE_LOG_D("OTAAgent::HardReboot");
    std::string rebootCMD = "reboot_orin";
    const std::string &sys_state_mgr_service_name = "tcp://localhost:11155";
    auto client_zmq = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_zmq->Init(sys_state_mgr_service_name);
    std::string reply;
    auto res = client_zmq->Request(rebootCMD, reply, 2000);
    if (res == 0 && reply == "success") {
        UPDATE_LOG_D("send [%s] command to sys_stat_mgr_service", rebootCMD.c_str());
    } else {
        UPDATE_LOG_D("send [%s] command failed !", rebootCMD.c_str());
        int res = system("echo 1 > /sys/class/tegra_hv_pm_ctl/tegra_hv_pm_ctl/device/trigger_sys_reboot");
        (void)res;
    }
    client_zmq->Deinit();
    return true;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
