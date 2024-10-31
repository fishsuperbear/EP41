#pragma once

#include <functional>
#include <mutex>
#include <memory>

#include "update_manager/common/data_def.h"
#include "update_manager/download/package_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class DownloadAgent {
public:

    static DownloadAgent* Instance();

    void Init();
    void Deinit();

    void RequestDownload(std::function<void(download_status_t*)> download_status_callback);
    void Verify(std::function<void(download_status_t*)> download_status_callback);

private:
    DownloadAgent();
    ~DownloadAgent();
    DownloadAgent(const DownloadAgent &);
    DownloadAgent & operator = (const DownloadAgent &);

    static std::mutex m_mtx;
    static DownloadAgent* m_pInstance;

    std::unique_ptr<PackageManager> pm_ptr_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
