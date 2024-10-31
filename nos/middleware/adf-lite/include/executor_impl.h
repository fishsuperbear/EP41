#include "adf-lite/include/executor.h"
#include "adf-lite/include/config.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ExecutorImpl {
public:
    ExecutorImpl();
    ~ExecutorImpl();
    void RegistAlgProcessFunc(const std::string& trigger, Executor::AlgProcessFunc func);
    void RegistAlgProcessWithProfilerFunc(const std::string& trigger, Executor::AlgProcessWithProfilerFunc func);
    Executor::AlgProcessFunc GetProcessFunc(const std::string& trigger);
    Executor::AlgProcessWithProfilerFunc GetProcessWithProfilerFunc(const std::string& trigger);
    void RegistPauseTriggerCb(Executor::PauseTriggerCb func);

    int32_t SendOutput(Bundle* output);
    int32_t SendOutput(Bundle* output, const ProfileToken& token);
    int32_t SendOutput(const std::string& topic, BaseDataTypePtr data);
    int32_t SendOutput(const std::string& topic, BaseDataTypePtr data, const ProfileToken& token);

    std::string GetConfigFilePath();
    void SetConfigFilePath(const std::string& path);
    ExecutorConfig* GetConfig();
    void SetConfig(ExecutorConfig* config);

    int32_t PauseTrigger(const std::string& trigger);
    int32_t ResumeTrigger(const std::string& trigger);
    int32_t PauseTrigger();
    int32_t ResumeTrigger();
private:
    Executor* _executor;
    std::unordered_map<std::string, Executor::AlgProcessFunc> _trigger_process_func_map;
    std::unordered_map<std::string, Executor::AlgProcessWithProfilerFunc> _trigger_process_with_profiler_func_map;
    ExecutorConfig* _config_ptr;
    std::string _config_file_path;
    Executor::PauseTriggerCb _pausetrigger_cb;
    std::mutex _regist_mutex;
    std::mutex _regist_mutex_with_profiler;
};

}    
}
}