#include "update_manager/agent/download_agent.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

DownloadAgent* DownloadAgent::m_pInstance = nullptr;
std::mutex DownloadAgent::m_mtx;


DownloadAgent::DownloadAgent()
{
    pm_ptr_ = std::make_unique<PackageManager>();
}

DownloadAgent::~DownloadAgent()
{
}

DownloadAgent*
DownloadAgent::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new DownloadAgent();
        }
    }

    return m_pInstance;
}

void
DownloadAgent::Init()
{
    UM_INFO << "DownloadAgent::Init.";
    UM_INFO << "DownloadAgent::Init Done.";
    // pm_ptr_-
}

void
DownloadAgent::Deinit()
{
    UM_INFO << "DownloadAgent::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "DownloadAgent::Deinit Done.";
}

void
DownloadAgent::RequestDownload(std::function<void(download_status_t*)> download_status_callback)
{
    pm_ptr_->RequestDownload(download_status_callback);
}

void
DownloadAgent::Verify(std::function<void(download_status_t*)> download_status_callback)
{
    pm_ptr_->Verify(download_status_callback);
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
