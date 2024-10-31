#pragma once

#include <stdint.h>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

class SensorsUdsManager {
public:

    static SensorsUdsManager* Instance();

    void Init();
    void Deinit();
    void Start();
    void Stop();
    bool GetVersion(const std::string& snesorName, std::string& snesorVersion);

private:
    bool PraseJsonFile();
    bool SendRequest(const SensorsDataInfo& info, std::string& version);
    bool ReleaseConnection(const std::string& snesorName);

private:
    SensorsUdsManager();
    ~SensorsUdsManager();
    SensorsUdsManager(const SensorsUdsManager &);
    SensorsUdsManager & operator = (const SensorsUdsManager &);

    static std::mutex m_mtx;
    static SensorsUdsManager* m_pInstance;

    std::mutex mutex_;
    std::condition_variable msg_received_cv_;
    std::atomic<bool> messageSuccess;
    std::atomic<bool> isStopped;

    std::vector<SensorsDataInfo> infos_{};
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
