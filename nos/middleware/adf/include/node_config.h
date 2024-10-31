#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "adf/include/optional.h"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace adf {
class NodeConfig {
   public:
    struct CommInstanceConfig {
        std::string name;
        std::string type;
        int domain;
        std::string topic;
        uint32_t buffer_capacity;
        bool is_async;
        bool freq_valid;
        double exp_freq;
        double freq_threshold_ratio;
    };

    struct ResourceLimit {
        bool valid;
        std::string group;
        uint32_t cpu;
        uint32_t memory;
    };

    struct Log {
        std::string name;
        std::string description;
        std::string file;
        uint32_t mode;
        uint32_t level;
        uint32_t adf_level;
    };

    struct Schedule {
        bool valid;
        int32_t policy;
        int32_t priority;
        cpu_set_t affinity;
    };

    struct Trigger {
        struct MainSource {
            std::string name;
            uint32_t timeout_ms;
        };

        struct AuxSource {
            std::string name;
            uint32_t multi_frame;
            bool read_clear;
        };

        struct Destination {
            std::string name;
        };

        std::string name;
        std::string type;
        uint32_t period_ms;
        uint32_t time_window_ms;
        uint32_t align_timeout_ms;
        uint32_t align_validity_ms;
        std::vector<MainSource> main_sources;
        std::vector<AuxSource> aux_sources;
        std::vector<Destination> destinations;
        Optional<uint64_t> exp_exec_time_ms;
    };

    struct ThreadPoolConfig {
        uint32_t num;
    };

    struct LatencyLink {
        std::string name;
        std::string recv_msg;
        std::string send_msg;
    };

    struct LatencyShow {
        std::string link_name;
        std::string trigger_name;
        std::string from_msg;
    };

    struct Latency {
        bool enable;
        std::vector<LatencyLink> links;
        std::vector<LatencyShow> shows;
    };

    struct CheckPoint {
        bool enable;
    };

    struct Profiler {
        std::string name;
        bool enable;
        Latency latency;
        CheckPoint checkpoint;
    };

    struct Monitor {
        bool freq_enable;
        bool latency_enable;
        uint64_t print_period_ms;
    };

    int32_t Parse(const std::string& file);

    std::vector<CommInstanceConfig> send_instance_configs;
    std::vector<CommInstanceConfig> recv_instance_configs;
    ResourceLimit resource_limit;
    Log log;
    Schedule schedule;
    std::vector<Trigger> trigger;
    ThreadPoolConfig thread_pool;
    Profiler profiler;
    Monitor monitor;

   private:
    int32_t ParseCommInstanceConfig(std::vector<CommInstanceConfig>& instances, const std::string& title);
    int32_t ParseResourceLimit();
    int32_t ParseLog();
    int32_t ParseSchedule();
    int32_t ParseTrigger();
    int32_t ParseThreadPool();
    int32_t ParseProfiler();
    int32_t ParseMonitor();
    void SetRecvInstanceAsync(const std::string& name, bool is_async);

    YAML::Node _node;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon