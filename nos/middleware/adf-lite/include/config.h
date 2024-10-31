#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sched.h>
#include "adf/include/optional.h"
#include "yaml-cpp/yaml.h"
#include "adf-lite/include/base.h"
#include "adf/include/node_config.h"

using namespace hozon::netaos::adf;

namespace hozon {
namespace netaos {
namespace adf_lite {


class TopConfig {
public:
    struct Log {
        std::string name;
        std::string description;
        std::string file;
        uint32_t mode;
        uint32_t level;
        uint32_t adfl_level;
    };

    struct ResourceLimit {
        std::string group;
        uint32_t cpu;
        uint32_t memory;
    };

    struct Schedule {
        int32_t policy;
        int32_t priority;
        cpu_set_t affinity;
    };

    struct ExecutorInfo {
        uint32_t order;
        std::string config_file;
    };

    std::string process_name;
    Log log;
    Optional<ResourceLimit> resource_limit;
    Optional<Schedule> schedule;
    std::vector<ExecutorInfo> executors;

    int32_t Parse(const std::string& file);

private:
    int32_t ParseProcessName();
    int32_t GetProcessName(std::string& value);
    int32_t ParseLog();
    int32_t ParseResourceLimit();
    int32_t ParseSchedule();
    int32_t ParseExecutor();
    YAML::Node _node;
    std::string _file;
};

class ExecutorConfig {
public:
    struct Log {
        std::string name;
        uint32_t level;
    };

    struct Input {
        std::string topic;
        uint32_t capacity;
    };

    struct Output {
        std::string topic;
    };

    struct Schedule {
        int32_t policy;
        int32_t priority;
        cpu_set_t affinity;
    };

    struct Trigger {
        enum class Type : uint32_t {
            EVENT = 0,
            PERIOD = 1,
            TS_ALIGN = 2,
            FREE = 3,
        };

        struct MainSource {
            std::string name;
            uint32_t timeout_ms;
        };

        struct AuxSource {
            std::string name;
            uint32_t multi_frame;
            bool read_clear;
        };


        std::string name;
        Type type;
        uint32_t period_ms;
        uint32_t time_window_ms;
        Optional<uint32_t> align_timeout_ms;
        uint32_t align_validity_ms;
        std::vector<MainSource> main_sources;
        std::vector<AuxSource> aux_sources;
        Optional<uint64_t> exp_exec_time_ms;

    };

    std::string library;
    std::vector<std::string> dep_lib_path;
    std::string executor_name;
    Log log;
    std::vector<Input> inputs;
    std::vector<Output> outputs;
    Optional<Schedule> schedule;
    std::vector<Trigger> triggers;
    NodeConfig::Profiler profiler;
    int32_t Parse(const std::string& file);

private:
    int32_t ParseFundamental();
    int32_t ParseLog();
    int32_t ParseProfiler();
    int32_t ParseInput();
    int32_t ParseOutput();
    int32_t ParseSchedule();
    int32_t ParseTrigger();

    YAML::Node _node;
    std::string _file;
};

}
}
}