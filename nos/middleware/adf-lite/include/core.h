#pragma once

#include "adf-lite/include/executor_mgr.h"
#include "adf-lite/include/config.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class Core {
public:
    Core();
    ~Core();

    int32_t Start(const std::string& top_config_file);
    void Stop();

private:
    int32_t InitLogger();
    int32_t InitScheduler();
    void StartInitConfig(const std::string& config_file);
    int32_t StartExecutor(const std::string& config_file);
    TopConfig _config;
    std::unordered_map<std::string, std::shared_ptr<ExecutorMgr>> _executor_mgr_map;
    std::vector<std::shared_ptr<std::thread>> _executor_thr;
};

}
}
}