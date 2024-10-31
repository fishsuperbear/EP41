#pragma once

#include <thread>
#include "adf/include/data_types/common/types.h"
#include "adf/include/internal_log.h"
#include "adf/include/node_base.h"
#include "adf/include/node_bundle.h"
#include "adf/include/node_comm.h"
#include "adf/include/node_config.h"
#include "adf/include/node_profiler_token.h"
#include "adf/include/profiler/checkpoint_profiler.h"
#include "adf/include/profiler/latency_profiler.h"
#include "adf/include/thread_pool.h"
#include "em/include/exec_client.h"
#include "phm/include/phm_client.h"

namespace hozon {
namespace netaos {
namespace adf {

#define DO_OR_ERROR_EARLY(statement, errortxt) \
    if ((statement) != 0) {                    \
        ADF_EARLY_LOG << errortxt;             \
        return -1;                             \
    }

class NodeBase;

class NodeBaseImpl {
   public:
    NodeBaseImpl(NodeBase* node_base);
    virtual ~NodeBaseImpl();

    int32_t Start(const std::string& config_file, bool log_already_inited);
    void Stop();
    static void SetNeedStop(bool need_stop);
    static bool NeedStop();
    static bool NeedStopBlocking();

    void Routine(NodeConfig::Trigger trigger, std::function<int32_t(NodeBundle* input)> alg_process_func,
                 std::function<int32_t(NodeBundle* input, const ProfileToken& token)> alg_process_with_profiler_func);

    void RegistAlgProcessFunc(const std::string& trigger, std::function<int32_t(NodeBundle* input)> func);
    void RegistAlgProcessWithProfilerFunc(const std::string& trigger,
                                          std::function<int32_t(NodeBundle* input, const ProfileToken& token)> func);

    std::shared_ptr<ThreadPool> GetThreadPool();

    int32_t SendOutput(NodeBundle* output);
    int32_t SendOutput(NodeBundle* output, const ProfileToken& token);

    const NodeConfig& GetConfig();

    std::vector<std::string> GetTriggerList();
    std::vector<std::string> GetAuxSourceList(const std::string& trigger_name);

    void BypassSend();
    void BypassRecv();

    int32_t PauseTrigger(const std::string& trigger);
    int32_t PauseTriggerAndJoin(const std::string& trigger);
    int32_t ResumeTrigger(const std::string& trigger);

    int RegisterCMType(const std::string& name, PubSubTypeBasePtr pub_sub_type);

    static int32_t InitLoggerStandAlone(const std::string& config_file);
    void ReportFault(uint32_t _faultId, uint8_t _faultObj);

   private:
    BaseDataTypePtr GetOneDataBlocking(const std::string& name, uint32_t blocktime_ms);
    BaseDataTypePtr GetOneData(const std::string& name, const uint32_t freshDataTime = UINT32_MAX);
    BaseDataTypePtr PopOneData(const std::string& name);
    std::vector<BaseDataTypePtr> GetNdata(const std::string& name, const uint32_t n);
    std::vector<BaseDataTypePtr> PopNdata(const std::string& name, const uint32_t n);
    int32_t SendOneData(const std::string& name, BaseDataTypePtr data);

    int32_t CreateCommRecvInstance(std::unordered_map<std::string, NodeCommRecvInstance>& instance_map,
                                   std::vector<NodeConfig::CommInstanceConfig> instance_configs);
    int32_t CreateCommSendInstance(std::unordered_map<std::string, NodeCommSendInstance>& instance_map,
                                   std::vector<NodeConfig::CommInstanceConfig> instance_configs);

    static int32_t InitLogger(NodeConfig& config);

    int32_t RecvMainSources(NodeBundle& input, NodeConfig::Trigger& trigger);
    int32_t RecvAuxSources(NodeBundle& input, NodeConfig::Trigger& trigger);
    int32_t RecvAlignedMainSources(NodeBundle& input, NodeConfig::Trigger& trigger);
    void RecvAlignMainSourceRoutine(NodeConfig::Trigger& trigger, NodeConfig::Trigger::MainSource& main_source);
    int32_t GetAlignedMainSources(NodeBundle& input, NodeConfig::Trigger& trigger);
    int32_t AlignSources(NodeBundle& input, NodeConfig::Trigger& trigger);
    void InitLatencyProfiler(NodeConfig::Trigger& trigger, std::vector<NodeConfig::LatencyShow>& latency_shows,
                             LatencyProfiler& lat_profiler);
    void CalcLatencyProfiler(NodeConfig::Trigger& trigger, std::vector<NodeConfig::LatencyShow>& latency_shows,
                             LatencyProfiler& lat_profiler, NodeBundle& input);
    void CptProfilerBegin(CheckpointProfiler& cpt_profiler);
    void CptProfilerEnd(CheckpointProfiler& cpt_profiler);
    int32_t InitScheduler();
    int32_t InitResourceLimit();

    int32_t InitThreadPool();

    void GenProfileToken(NodeBundle& input, ProfileToken& token);

    void CheckForPause(const std::string& trigger);
    int32_t PauseSources(const std::string& trigger, const std::string& trigger_sources);
    int32_t ResumeSources(const std::string& trigger_sources);

    void InitMonitor();

    struct TriggerControl {
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

    NodeBase* _node_base;
#ifndef BUILD_FOR_X86
    em::ExecClient _exec_client;
#endif
    std::unordered_map<std::string, NodeCommRecvInstance> _recv_instances_map;
    std::unordered_map<std::string, NodeCommSendInstance> _send_instances_map;
    NodeConfig _config;
    std::vector<std::shared_ptr<std::thread>> _threads;
    bool _need_stop = true;
    static bool _term_signal;
    static std::condition_variable _term_cv;
    static std::mutex _term_mutex;

    std::shared_ptr<std::thread> _alive_report_thread;
    bool _bypass_send = false;
    bool _bypass_recv = false;
    std::unordered_map<std::string, std::function<int32_t(NodeBundle* input)>> _trigger_process_func_map;
    std::unordered_map<std::string, std::function<int32_t(NodeBundle* input, const ProfileToken& token)>>
        _trigger_process_with_profiler_func_map;
    std::unordered_map<std::string, TriggerControl> _trigger_control_map;
    std::shared_ptr<ThreadPool> _thread_pool = nullptr;
    std::shared_ptr<ThreadPool> _data_recv_thread_pool = nullptr;

    std::unique_ptr<phm::PHMClient> phm_client_;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
