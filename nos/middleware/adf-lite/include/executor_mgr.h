#pragma once

#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <thread>
#include <list>
#include "adf-lite/include/bundle.h"
#include "adf-lite/include/executor.h"
#include "adf-lite/include/writer.h"
#include "adf-lite/include/reader.h"
#include "adf-lite/include/config.h"
#include "adf/include/thread_pool.h"
#include "adf/include/profiler/profiler.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

struct TriggerControl {
    std::shared_ptr<std::thread> process_thread;

    bool pause_enable;
    bool pause_ready;
    std::mutex pause_mutex;
    std::condition_variable pause_ready_cv;
    std::condition_variable resume_cv;

    // for align
    std::mutex align_mutex;
    std::condition_variable align_data_arrived_cv;
    std::list<std::pair<std::string, BaseDataTypePtr>> align_list;
    std::unordered_map<std::string, uint64_t> source_latest_timestamp;
    std::vector<std::shared_ptr<std::thread>> align_recv_threads;
};

class ExecutorMgr {
public:
    ExecutorMgr();
    virtual ~ExecutorMgr();
    int32_t InitConfig(const std::string& config_file);
    int32_t Init();
    void StartProcess();
    void Stop();
    void PreStop();
private:
    friend class DbgInfo;
    int32_t LoadExecutor();
    int32_t InitScheduler();
    int32_t InitThreadPool();
    int32_t InitCommunication();

    void CheckForPause(const std::string& trigger);
    void Routine(ExecutorConfig::Trigger& trigger);
    int32_t RecvMainSources(Bundle& input, ExecutorConfig::Trigger& trigger);
    int32_t RecvAuxSources(Bundle& input, ExecutorConfig::Trigger& trigger);
    int32_t RecvAlignedMainSources(Bundle& input, ExecutorConfig::Trigger& trigger);
    void RecvAlignMainSourceRoutine(ExecutorConfig::Trigger& trigger, ExecutorConfig::Trigger::MainSource& source);
    int32_t GetAlignedMainSources(Bundle& input, ExecutorConfig::Trigger& trigger);
    int32_t AlignSources(Bundle& input, ExecutorConfig::Trigger& trigger);
    int32_t CheckDataAligned(ExecutorConfig::Trigger& trigger, std::unordered_map<std::string, BaseDataTypePtr>& data_map);
    int32_t PauseTrigger(const std::string& trigger, const bool pause);
    int32_t PauseTriggerPtr(const std::string& trigger, hozon::netaos::adf_lite::ExecutorConfig::Trigger* trigger_ptr, const bool pause);
    void GenProfileToken(Bundle& input, ProfileToken& token);
    void InitLatencyProfiler(
        ExecutorConfig::Trigger& trigger,
        std::vector<NodeConfig::LatencyShow>& latency_shows,
        LatencyProfiler& lat_profiler);
    void CalcLatencyProfiler(
        ExecutorConfig::Trigger& trigger,
        std::vector<NodeConfig::LatencyShow>& latency_shows,
        LatencyProfiler& lat_profiler,
        Bundle& input,
        bool after_process);
    void CptProfilerBegin(CheckpointProfiler& cpt_profiler);
    void CptProfilerEnd(CheckpointProfiler& cpt_profiler);

    LibraryLoader _executor_loader;
    std::shared_ptr<Executor> _executor;
    ExecutorConfig _config;
    std::shared_ptr<hozon::netaos::adf::ThreadPool> _data_recv_thread_pool = nullptr;
    bool _need_stop = false;
    // std::unordered_map<std::string, std::shared_ptr<Writer>> _writers;
    std::unordered_map<std::string, std::shared_ptr<Reader>> _readers;
    std::unordered_map<std::string, TriggerControl> _trigger_control_map;
    std::string _config_file_path;
};

}
}
}