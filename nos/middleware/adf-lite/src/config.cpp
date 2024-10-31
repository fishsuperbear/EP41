
#include <unistd.h>
#include "adf-lite/include/config.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "adf-lite/service/rpc/lite_rpc.h"
namespace hozon {
namespace netaos {
namespace adf_lite {

#ifdef DO
#undef DO
#endif

#ifdef DO_OR_ERROR
#undef DO_OR_ERROR
#endif

#define DO(statement) \
    if ((statement) < 0) { \
        return -1; \
    }

#define DO_OR_ERROR(statement, str, file) \
    if (!(statement)) { \
        ADF_EARLY_LOG << "Cannot find " << (str) << " in config file " << file; \
        return -1; \
    }

std::string RealPath(const std::string& path) {
    std::string cwd = get_current_dir_name();
    std::string target_path;
    if (path.substr(0, 2) == "./" || path.substr(0, 3) == "../") {
        target_path = cwd + "/" + path;
    } else if (path.substr(0, 21) == "${ADFLITE_ROOT_PATH}/" || path.substr(0, 19) == "$ADFLITE_ROOT_PATH/") {
        const char *adflite_root_path = getenv("ADFLITE_ROOT_PATH");
        if (adflite_root_path) {
            target_path = std::string(adflite_root_path) + path.substr(path.find_first_of('/'));
        } else {
            ADF_EARLY_LOG << "Error: The path [" << path << "] cannot generate successfully, because ADFLITE_ROOT_PATH has not been set. ";
            target_path = path.substr(path.find_first_of('/'));
        }
    } else {
        return path;
    }
    return target_path;
}

int32_t TopConfig::ParseProcessName() {
    DO_OR_ERROR(_node["processName"], "processName", _file);
    process_name = _node["processName"].as<std::string>();

    ADF_EARLY_LOG << "process_name is " << process_name;
    return 0;
}

int32_t TopConfig::ParseLog() {
    DO_OR_ERROR(_node["appLog"], "appLog", _file);
    DO_OR_ERROR(_node["appLog"]["name"], "appLog.name", _file);
    DO_OR_ERROR(_node["appLog"]["description"], "appLog.description", _file);
    // DO_OR_ERROR(_node["appLog"]["file"], "appLog.file", _file);
    DO_OR_ERROR(_node["appLog"]["mode"], "appLog.mode", _file);
    DO_OR_ERROR(_node["appLog"]["level"], "appLog.level", _file);

    log.name = _node["appLog"]["name"].as<std::string>();
    log.description = _node["appLog"]["description"].as<std::string>();
    if (_node["appLog"]["file"]) {
        log.file = _node["appLog"]["file"].as<std::string>();
    }
    else {
        log.file = "./";
    }
    
    log.mode = _node["appLog"]["mode"].as<uint32_t>();
    if (log.mode < 0 || log.mode > 6) {
        log.mode = 0;
    }
    log.level = _node["appLog"]["level"].as<uint32_t>();
    if (log.level < 0 || log.level > 6) {
        log.level = 6;
    }
    if (_node["appLog"]["adfl_level"]) {
        log.adfl_level = _node["appLog"]["adfl_level"].as<uint32_t>();
    } else {
        log.adfl_level = log.level;
    }
    return 0;
}

int32_t TopConfig::ParseResourceLimit() {
    if (!_node["resourceLimit"]) {
        return 0;
    }

    DO_OR_ERROR(_node["resourceLimit"]["group"], "resourceLimit.group", _file);
    DO_OR_ERROR(_node["resourceLimit"]["cpu"], "resourceLimit.cpu", _file);
    DO_OR_ERROR(_node["resourceLimit"]["memory"], "resourceLimit.memory", _file);

    resource_limit.Value().group = _node["resourceLimit"]["group"].as<std::string>();
    resource_limit.Value().cpu = _node["resourceLimit"]["cpu"].as<uint32_t>();
    resource_limit.Value().memory = _node["resourceLimit"]["memory"].as<uint32_t>();

    return 0;
}

int32_t TopConfig::ParseSchedule() {
    if (!_node["schedule"]) {
        return 0;
    }

    schedule.Value().policy = SCHED_OTHER;
    schedule.Value().priority = 0;
    CPU_ZERO(&schedule.Value().affinity);
    for (std::size_t i = 0; i < 16; ++i) {
        CPU_SET(i, &schedule.Value().affinity);
    }

    if (_node["schedule"]["policy"]) {
        std::string policy = _node["schedule"]["policy"].as<std::string>();
        if (policy == "RR") {
            schedule.Value().policy = SCHED_RR;
        }
        else if (policy == "FIFO") {
            schedule.Value().policy = SCHED_FIFO;
        }
        else if (policy == "OTHER") {
            schedule.Value().policy = SCHED_OTHER;
        }
        else {
            ADF_EARLY_LOG << "Unsupported scheduler policy " << policy;
            return -1;
        }
    }

    if (_node["schedule"]["priority"]) {
        schedule.Value().priority = _node["schedule"]["priority"].as<uint32_t>();
        if (schedule.Value().priority < 0 || schedule.Value().priority > 99) {
            ADF_EARLY_LOG << "schedule.priority should be from 0 to 99.";
            return -1;
        }
    }

    if (_node["schedule"]["cpuAffinity"].IsSequence()) {
        CPU_ZERO(&schedule.Value().affinity);
        for (std::size_t i = 0; i < _node["schedule"]["cpuAffinity"].size(); ++i) {
            CPU_SET(_node["schedule"]["cpuAffinity"][i].as<uint32_t>(), &schedule.Value().affinity);
        }
    }

    return 0;
}

int32_t TopConfig::ParseExecutor() {
    DO_OR_ERROR(_node["executors"], "executors", _file);

    for (size_t i = 0; i < _node["executors"].size(); ++i) {
        ExecutorInfo executor_info;

        DO_OR_ERROR(_node["executors"][i]["confFile"], "executors.confFile", _file);
        if (_node["executors"][i]["order"]) {
            executor_info.order = _node["executors"][i]["order"].as<uint32_t>();
        } else {
            executor_info.order = 100;
        }
        executor_info.config_file = RealPath(_node["executors"][i]["confFile"].as<std::string>());
        executors.emplace_back(executor_info);
    }

    return 0;
}

int32_t TopConfig::GetProcessName(std::string& value)
{
    value = process_name;
    return 0;
}


int32_t TopConfig::Parse(const std::string& file) {
    _file = file;
    _node = YAML::LoadFile(file);
    if (!_node) {
        ADF_EARLY_LOG << "Fail to load config file " << file;
        return -1;
    }
    DO(ParseProcessName());
    DO(ParseLog());
    // DO(ParseSchedule());
    DO(ParseResourceLimit());
    DO(ParseExecutor());

    LiteRpc::GetInstance().RegisterStringServiceFunc("GetProcessName", std::bind(&TopConfig::GetProcessName, this, std::placeholders::_1));
    return 0;
}

int32_t ExecutorConfig::ParseFundamental() {
    DO_OR_ERROR(_node["library"], "library", _file);
    // DO_OR_ERROR(_node["depLibPath"], "depLibPath", _file);
    DO_OR_ERROR(_node["executorName"], "executorName", _file);

    library = RealPath(_node["library"].as<std::string>());
    if (_node["depLibPath"]) {
        for (size_t i = 0; i < _node["depLibPath"].size(); ++i) {
            dep_lib_path.emplace_back(_node["depLibPath"][i].as<std::string>());
        }
    }
    executor_name = _node["executorName"].as<std::string>();

    return 0;
}

int32_t ExecutorConfig::ParseLog() {
    DO_OR_ERROR(_node["log"], "log", _file);
    DO_OR_ERROR(_node["log"]["name"], "log.name", _file);
    DO_OR_ERROR(_node["log"]["level"], "log.level", _file);

    log.name = _node["log"]["name"].as<std::string>();
    log.level = _node["log"]["level"].as<uint32_t>();
    if (log.level < 0 || log.level > 6) {
        log.level = 6;
    }
    return 0;
}

int32_t ExecutorConfig::ParseInput() {
    // DO_OR_ERROR(_node["input"], "input", _file);
    if (!_node["input"]) {
        return 0;
    }

    for (size_t i = 0; i < _node["input"].size(); ++i) {
        Input input;
        DO_OR_ERROR(_node["input"][i]["topic"], "input.topic", _file);
        input.topic = _node["input"][i]["topic"].as<std::string>();

        if (_node["input"][i]["capacity"]) {
            input.capacity = _node["input"][i]["capacity"].as<uint32_t>();
        }
        else {
            input.capacity = 5;
        }
        
        inputs.emplace_back(input);
    }
    
    return 0;
}

int32_t ExecutorConfig::ParseOutput() {
    // DO_OR_ERROR(_node["output"], "output", _file);
    if (!_node["output"]) {
        return 0;
    }
    for (size_t i = 0; i < _node["output"].size(); ++i) {
        Output output;
        DO_OR_ERROR(_node["output"][i]["topic"], "output.topic", _file);
        output.topic = _node["output"][i]["topic"].as<std::string>();
        
        outputs.emplace_back(output);
    }
    
    return 0;
}

int32_t ExecutorConfig::ParseSchedule() {
    if (!_node["schedule"]) {
        return 0;
    }

    schedule.Value().policy = SCHED_OTHER;
    schedule.Value().priority = 0;
    CPU_ZERO(&schedule.Value().affinity);
    for (std::size_t i = 0; i < 16; ++i) {
        CPU_SET(i, &schedule.Value().affinity);
    }

    if (_node["schedule"]["policy"]) {
        std::string policy = _node["schedule"]["policy"].as<std::string>();
        if (policy == "RR") {
            schedule.Value().policy = SCHED_RR;
        }
        else if (policy == "FIFO") {
            schedule.Value().policy = SCHED_FIFO;
        }
        else if (policy == "OTHER") {
            schedule.Value().policy = SCHED_OTHER;
        }
        else {
            ADF_EARLY_LOG << "Unsupported scheduler policy " << policy;
            return -1;
        }
    }

    if (_node["schedule"]["priority"]) {
        schedule.Value().priority = _node["schedule"]["priority"].as<uint32_t>();
        if (schedule.Value().priority < 0 || schedule.Value().priority > 99) {
            ADF_EARLY_LOG << "schedule.priority should be from 0 to 99.";
            return -1;
        }
    }

    if (_node["schedule"]["cpuAffinity"].IsSequence()) {
        CPU_ZERO(&schedule.Value().affinity);
        for (std::size_t i = 0; i < _node["schedule"]["cpuAffinity"].size(); ++i) {
            CPU_SET(_node["schedule"]["cpuAffinity"][i].as<uint32_t>(), &schedule.Value().affinity);
        }
    }

    return 0;
}

int32_t ExecutorConfig::ParseTrigger() {
    // DO_OR_ERROR(_node["trigger"].IsSequence(), "trigger", _file);
    if (!_node["trigger"]) {
        return 0;
    }
    
    for (std::size_t k = 0; k < _node["trigger"].size(); ++k) {
        Trigger tmp;

        DO_OR_ERROR(_node["trigger"][k]["name"], "trigger.name", _file);
        tmp.name = _node["trigger"][k]["name"].as<std::string>();

        DO_OR_ERROR(_node["trigger"][k]["type"], "trigger.type", _file);
        std::string type;
        type = _node["trigger"][k]["type"].as<std::string>();
         
        if (_node["trigger"][k]["period"]) {
            tmp.period_ms = _node["trigger"][k]["period"].as<uint32_t>();
        }
        else {
            tmp.period_ms = 0;
        }

        if (type == "EVENT") {
            tmp.type = ExecutorConfig::Trigger::Type::EVENT;
            DO_OR_ERROR(_node["trigger"][k]["mainSources"].IsSequence(), "trigger.mainSources", _file);
            for (std::size_t i = 0; i < _node["trigger"][k]["mainSources"].size(); ++i) {
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["name"], "trigger.mainSources.name", _file);
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["timeout"], "trigger.mainSources.timeout", _file);
                Trigger::MainSource main_source;
                main_source.name = _node["trigger"][k]["mainSources"][i]["name"].as<std::string>();
                main_source.timeout_ms = _node["trigger"][k]["mainSources"][i]["timeout"].as<uint32_t>();

                tmp.main_sources.emplace_back(main_source);
            }
        }
        else if (type == "PERIOD") {
            tmp.type = ExecutorConfig::Trigger::Type::PERIOD;
            DO_OR_ERROR(tmp.period_ms != 0, "trigger.period", _file);
        }
        else if (type == "TS_ALIGN") {
            tmp.type = ExecutorConfig::Trigger::Type::TS_ALIGN;
            DO_OR_ERROR(_node["trigger"][k]["mainSources"].IsSequence(), "trigger.mainSources", _file);
            for (std::size_t i = 0; i < _node["trigger"][k]["mainSources"].size(); ++i) {
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["name"], "trigger.mainSources.name", _file);
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["timeout"], "trigger.mainSources.timeout", _file);
                Trigger::MainSource main_source;
                main_source.name = _node["trigger"][k]["mainSources"][i]["name"].as<std::string>();
                main_source.timeout_ms = _node["trigger"][k]["mainSources"][i]["timeout"].as<uint32_t>();

                tmp.main_sources.emplace_back(main_source);
            }

            DO_OR_ERROR(_node["trigger"][k]["timeWindow"], "trigger.timeWindow", _file);
            tmp.time_window_ms = _node["trigger"][k]["timeWindow"].as<uint32_t>();

            if (_node["trigger"][k]["alignTimeoutMs"]) {
                tmp.align_timeout_ms = _node["trigger"][k]["alignTimeoutMs"].as<uint32_t>();
            }

            if (_node["trigger"][k]["alignValidityMs"]) {
                tmp.align_validity_ms = _node["trigger"][k]["alignValidityMs"].as<uint32_t>();
            }
            else {
                tmp.align_validity_ms = 2 * tmp.time_window_ms;
            }
        }
        else if (type == "FREE") {
            tmp.type = ExecutorConfig::Trigger::Type::FREE;
        }
        else {
            ADF_EARLY_LOG << "Unsupported trigger type " << type;
            return -1;
        }

        if (_node["trigger"][k]["auxSources"].IsSequence()) {
            for (std::size_t i = 0; i < _node["trigger"][k]["auxSources"].size(); ++i) {
                Trigger::AuxSource aux_source;
                
                DO_OR_ERROR(_node["trigger"][k]["auxSources"][i]["name"], "trigger.auxSources.name", _file);
                aux_source.name = _node["trigger"][k]["auxSources"][i]["name"].as<std::string>();

                if (_node["trigger"][k]["auxSources"][i]["multiFrame"]) {
                    aux_source.multi_frame = _node["trigger"][k]["auxSources"][i]["multiFrame"].as<uint32_t>();
                }
                else {
                    aux_source.multi_frame = 0;
                }

                if (_node["trigger"][k]["auxSources"][i]["readClear"]) {
                    aux_source.read_clear = _node["trigger"][k]["auxSources"][i]["readClear"].as<bool>();
                }
                else {
                    aux_source.read_clear = false;
                }

                tmp.aux_sources.emplace_back(aux_source);
            }
        }

        if (_node["trigger"][k]["expExecTimeMs"]) {
            tmp.exp_exec_time_ms = _node["trigger"][k]["expExecTimeMs"].as<uint64_t>();
        }

        triggers.emplace_back(tmp);
    }
    
    return 0;
}

int32_t ExecutorConfig::ParseProfiler() {
    if (!_node["profiler"]) {
        profiler.enable = false;
        return 0;
    }

    DO_OR_ERROR(_node["profiler"]["name"], "profiler.name", _file);
    profiler.name = _node["profiler"]["name"].as<std::string>();

    DO_OR_ERROR(_node["profiler"]["enable"], "profiler.enable", _file);
    profiler.enable = _node["profiler"]["enable"].as<bool>();

    if (profiler.enable) {
        ADF_EARLY_LOG << "profiler.enable is " << profiler.enable;
    } else {
        ADF_EARLY_LOG << "profiler.enable is false " << profiler.enable;
    }
    if (!_node["profiler"]["latency"]) {
        profiler.latency.enable = false;
    }
    else {
        DO_OR_ERROR(_node["profiler"]["latency"]["enable"], "profiler.latency.enable", _file);
        profiler.latency.enable = _node["profiler"]["latency"]["enable"].as<bool>();

        if (_node["profiler"]["latency"]["link"]) {
            for (std::size_t i = 0; i < _node["profiler"]["latency"]["link"].size(); ++i) {
                NodeConfig::LatencyLink link;

                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["name"], "profiler.latency.link[].name", _file);
                link.name = _node["profiler"]["latency"]["link"][i]["name"].as<std::string>();
                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["recvMsg"], "profiler.latency.link[].recvMsg", _file);
                link.recv_msg = _node["profiler"]["latency"]["link"][i]["recvMsg"].as<std::string>();
                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["sendMsg"], "profiler.latency.link[].sendMsg", _file);
                link.send_msg = _node["profiler"]["latency"]["link"][i]["sendMsg"].as<std::string>();
                profiler.latency.links.emplace_back(link);
            }
        }
        ADF_EARLY_LOG << "parse profiler.latency.show";
        if (_node["profiler"]["latency"]["show"]) {
            for (std::size_t i = 0; i < _node["profiler"]["latency"]["show"].size(); ++i) {
                ADF_EARLY_LOG << "has profiler.latency.show i= " << i;
                NodeConfig::LatencyShow show;

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["link_name"], "profiler.latency.show[].link_name", _file);
                show.link_name = _node["profiler"]["latency"]["show"][i]["link_name"].as<std::string>();

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["trigger_name"], "profiler.latency.show[].trigger_name", _file);
                show.trigger_name = _node["profiler"]["latency"]["show"][i]["trigger_name"].as<std::string>();

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["from_msg"], "profiler.latency.show[].from_msg", _file);
                show.from_msg = _node["profiler"]["latency"]["show"][i]["from_msg"].as<std::string>();
                profiler.latency.shows.emplace_back(show);
            }
        } else {
            ADF_EARLY_LOG << "has not profiler.latency.show";
        }
    }

    if (!_node["profiler"]["checkpoint"]) {
        profiler.checkpoint.enable = false;
    }
    else {
        DO_OR_ERROR(_node["profiler"]["checkpoint"]["enable"], "profiler.checkpoint.enable", _file);
        profiler.checkpoint.enable = _node["profiler"]["checkpoint"]["enable"].as<bool>();
    }

    return 0;
}

int32_t ExecutorConfig::Parse(const std::string& file) {
    _file = file;
    _node = YAML::LoadFile(file);
    if (!_node) {
        ADF_EARLY_LOG << "Fail to load config file " << file;
        return -1;
    }

    DO(ParseFundamental());
    // DO(ParseLog());
    DO(ParseProfiler());
    DO(ParseInput());
    DO(ParseOutput());
    // DO(ParseSchedule());
    DO(ParseTrigger());

    return 0;
}

}
}
}