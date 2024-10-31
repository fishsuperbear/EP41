#include "adf-lite/include/executor_impl.h"
#include "adf-lite/include/writer.h"
#include "adf-lite/include/adf_lite_internal_logger.h"


namespace hozon {
namespace netaos {
namespace adf_lite {

ExecutorImpl::ExecutorImpl() {
    (void)_executor;
}

ExecutorImpl::~ExecutorImpl() {

}

void ExecutorImpl::RegistAlgProcessFunc(const std::string& trigger, Executor::AlgProcessFunc func) {
    std::lock_guard<std::mutex> regist_lk(_regist_mutex);
    _trigger_process_func_map[trigger] = func;
}

void ExecutorImpl::RegistAlgProcessWithProfilerFunc(const std::string& trigger, Executor::AlgProcessWithProfilerFunc func) {
    std::lock_guard<std::mutex> regist_lk(_regist_mutex_with_profiler);
    ADF_INTERNAL_LOG_INFO << "RegistAlgProcessWithProfilerFunc " << trigger;
    _trigger_process_with_profiler_func_map[trigger] = func;
}


Executor::AlgProcessFunc ExecutorImpl::GetProcessFunc(const std::string& trigger) {
    std::lock_guard<std::mutex> regist_lk(_regist_mutex);
    if (_trigger_process_func_map.find(trigger) == _trigger_process_func_map.end()) {
        ADF_INTERNAL_LOG_ERROR << trigger << " process func is not found in _trigger_process_func_map";
        return nullptr;
    }

    return _trigger_process_func_map[trigger];
}

Executor::AlgProcessWithProfilerFunc ExecutorImpl::GetProcessWithProfilerFunc(const std::string& trigger) {
    std::lock_guard<std::mutex> regist_lk(_regist_mutex_with_profiler);
    if (_trigger_process_with_profiler_func_map.find(trigger) == _trigger_process_with_profiler_func_map.end()) {
        ADF_INTERNAL_LOG_ERROR << trigger << " process func is not found in _trigger_process_with_profiler_func_map";
        return nullptr;
    }

    return _trigger_process_with_profiler_func_map[trigger];
}

void ExecutorImpl::RegistPauseTriggerCb(Executor::PauseTriggerCb func) {
    _pausetrigger_cb = func;
}

int32_t ExecutorImpl::SendOutput(Bundle* output) {
    auto& raw_data = output->GetRaw();

    for (auto& topic_data_vec : raw_data) {
        for (auto& data : topic_data_vec.second) {
            SendOutput(topic_data_vec.first, data);
        }
    }

    return 0;
}

int32_t ExecutorImpl::SendOutput(Bundle* output, const ProfileToken& token) {
    if (GetConfig() != nullptr) {
        if (GetConfig()->profiler.enable && GetConfig()->profiler.latency.enable) {
            for (auto link : GetConfig()->profiler.latency.links) {
                auto send_msgs = output->GetAll(link.send_msg);
                if (send_msgs.empty()) {
                    ADF_INTERNAL_LOG_WARN << "Empty send msg [" << link.send_msg << "], bypass profile.";
                    continue;
                }
                else {
                    for (auto send_msg : send_msgs) {
                        send_msg->__header.latency_info = token.latency_info;
                    }
                }
            }
        }
    }

    return SendOutput(output);
}

int32_t ExecutorImpl::SendOutput(const std::string& topic, BaseDataTypePtr data) {
    Writer _writer;
    _writer.Init(topic);
    _writer.Write(data);
    return 0;
}

int32_t ExecutorImpl::SendOutput(const std::string& topic, BaseDataTypePtr data, const ProfileToken& token) {
    if (GetConfig() != nullptr) {
        if (GetConfig()->profiler.enable && GetConfig()->profiler.latency.enable) {
            for (auto link : GetConfig()->profiler.latency.links) {
                if (topic == link.send_msg) {
                    data->__header.latency_info = token.latency_info;
                }
            }
        }
    }
    SendOutput(topic, data);
    return 0;
}

int32_t ExecutorImpl::PauseTrigger(const std::string& trigger) {
    if (_pausetrigger_cb != nullptr) {
        _pausetrigger_cb(trigger, true);
        return 0;
    } else {
        ADF_INTERNAL_LOG_ERROR << "_pausetrigger_cb is nullptr";
        return -1;
    }
}

int32_t ExecutorImpl::ResumeTrigger(const std::string& trigger) {
    if (_pausetrigger_cb != nullptr) {
        _pausetrigger_cb(trigger, false);
        return 0;
    } else {
        ADF_INTERNAL_LOG_ERROR << "_pausetrigger_cb is nullptr";
        return -1;
    }
}

int32_t ExecutorImpl::PauseTrigger() {
    if (_pausetrigger_cb != nullptr) {
        _pausetrigger_cb(ALLTRIGGER_NAME, true);
        return 0;
    } else {
        ADF_INTERNAL_LOG_ERROR << "_pausetrigger_cb is nullptr";
        return -1;
    }
}

int32_t ExecutorImpl::ResumeTrigger() {
    if (_pausetrigger_cb != nullptr) {
        _pausetrigger_cb(ALLTRIGGER_NAME, false);
        return 0;
    } else {
        ADF_INTERNAL_LOG_ERROR << "_pausetrigger_cb is nullptr";
        return -1;
    }
}

std::string ExecutorImpl::GetConfigFilePath() {
    return _config_file_path;
}

void ExecutorImpl::SetConfigFilePath(const std::string& path) {
    _config_file_path = path;
}

ExecutorConfig* ExecutorImpl::GetConfig() {
    return _config_ptr;
}
void ExecutorImpl::SetConfig(ExecutorConfig* config) {
    _config_ptr = config;
}

}
}
}

