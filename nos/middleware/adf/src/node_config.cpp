#include "adf/include/node_config.h"
#include "adf/include/internal_log.h"

namespace hozon {
namespace netaos {
namespace adf {
#define DO(statement)      \
    if ((statement) < 0) { \
        return -1;         \
    }

#define DO_OR_ERROR(statement, str)                                     \
    if (!(statement)) {                                                 \
        ADF_EARLY_LOG << "Cannot find " << (str) << " in config file."; \
        return -1;                                                      \
    }

int32_t NodeConfig::Parse(const std::string& file) {
    _node = YAML::LoadFile(file);
    if (!_node) {
        ADF_EARLY_LOG << "Fail to load config file " << file;
        return -1;
    }

    DO(ParseCommInstanceConfig(send_instance_configs, "sendInstances"));
    DO(ParseCommInstanceConfig(recv_instance_configs, "recvInstances"));

    ParseResourceLimit();
    DO(ParseLog());
    ParseSchedule();
    DO(ParseTrigger());
    ParseThreadPool();
    ParseProfiler();
    ParseMonitor();
    return 0;
}

int32_t NodeConfig::ParseCommInstanceConfig(std::vector<CommInstanceConfig>& instances, const std::string& title) {
    if (!_node[title].IsSequence()) {
        ADF_EARLY_LOG << "No " << title << " specified.";
        return 0;
    }

    for (std::size_t i = 0; i < _node[title].size(); ++i) {
        CommInstanceConfig instance;
        DO_OR_ERROR(_node[title][i]["name"], title + ".name");
        instance.name = _node[title][i]["name"].as<std::string>();

        DO_OR_ERROR(_node[title][i]["type"], title + ".type");
        instance.type = _node[title][i]["type"].as<std::string>();

        DO_OR_ERROR(_node[title][i]["topic"], title + ".topic");
        instance.topic = _node[title][i]["topic"].as<std::string>();

        DO_OR_ERROR(_node[title][i]["domainId"], title + ".domainId");
        instance.domain = _node[title][i]["domainId"].as<uint32_t>();

        if (_node[title][i]["bufferCapacity"]) {
            instance.buffer_capacity = _node[title][i]["bufferCapacity"].as<uint32_t>();
        } else {
            instance.buffer_capacity = 5;
        }
        if (_node[title][i]["is_async"]) {
            instance.is_async = _node[title][i]["is_async"].as<bool>();
            // ADF_EARLY_LOG << "Comm instance, is_async: " << instance.is_async;
        } else {
            instance.is_async = true;
            ;
        }

        if (_node[title][i]["expFreq"] && _node[title][i]["freqThresholdRatio"]) {
            DO_OR_ERROR(_node[title][i]["expFreq"], title + ".expFreq");
            instance.exp_freq = _node[title][i]["expFreq"].as<double>();

            DO_OR_ERROR(_node[title][i]["freqThresholdRatio"], title + ".freqThresholdRatio");
            instance.freq_threshold_ratio = _node[title][i]["freqThresholdRatio"].as<double>();

            instance.freq_valid = true;
        } else {
            instance.freq_valid = false;
        }

        instances.emplace_back(instance);
    }

    return 0;
}

int32_t NodeConfig::ParseResourceLimit() {
    resource_limit.valid = false;
    DO_OR_ERROR(_node["resourceLimit"], "resourceLimit");
    DO_OR_ERROR(_node["resourceLimit"]["group"], "resourceLimit.group");
    DO_OR_ERROR(_node["resourceLimit"]["cpu"], "resourceLimit.cpu");
    DO_OR_ERROR(_node["resourceLimit"]["memory"], "resourceLimit.memory");

    resource_limit.group = _node["resourceLimit"]["group"].as<std::string>();
    resource_limit.cpu = _node["resourceLimit"]["cpu"].as<uint32_t>();
    resource_limit.memory = _node["resourceLimit"]["memory"].as<uint32_t>();
    resource_limit.valid = true;

    return 0;
}

int32_t NodeConfig::ParseLog() {
    DO_OR_ERROR(_node["log"], "log");
    DO_OR_ERROR(_node["log"]["mode"], "log.mode");
    DO_OR_ERROR(_node["log"]["name"], "log.name");
    DO_OR_ERROR(_node["log"]["description"], "log.description");
    DO_OR_ERROR(_node["log"]["level"], "log.level");

    log.mode = _node["log"]["mode"].as<uint32_t>();
    if (_node["log"]["file"]) {
        log.file = _node["log"]["file"].as<std::string>();
    }

    log.name = _node["log"]["name"].as<std::string>();
    log.description = _node["log"]["description"].as<std::string>();
    log.level = _node["log"]["level"].as<uint32_t>();
    if (_node["log"]["adf"]) {
        DO_OR_ERROR(_node["log"]["adf"]["level"], "log.adflog.level");
        log.adf_level = _node["log"]["adf"]["level"].as<uint32_t>();
    } else {
        log.adf_level = 6;
    }
    return 0;
}

int32_t NodeConfig::ParseSchedule() {
    schedule.valid = false;
    schedule.policy = SCHED_OTHER;
    schedule.priority = 0;
    CPU_ZERO(&schedule.affinity);
    for (std::size_t i = 0; i < 16; ++i) {
        CPU_SET(i, &schedule.affinity);
    }

    if (!_node["schedule"]) {
        return 0;
    }

    if (_node["schedule"]["policy"]) {
        std::string policy = _node["schedule"]["policy"].as<std::string>();
        if (policy == "RR") {
            schedule.policy = SCHED_RR;
        } else if (policy == "FIFO") {
            schedule.policy = SCHED_FIFO;
        } else if (policy == "OTHER") {
            schedule.policy = SCHED_OTHER;
        } else {
            ADF_EARLY_LOG << "Unsupported scheduler policy " << policy;
            return -1;
        }
    }

    if (_node["schedule"]["priority"]) {
        schedule.priority = _node["schedule"]["priority"].as<uint32_t>();
        if (schedule.priority < 0 || schedule.priority > 99) {
            ADF_EARLY_LOG << "schedule.priority should be from 0 to 99.";
            return -1;
        }
    }

    if (_node["schedule"]["cpuAffinity"].IsSequence()) {
        CPU_ZERO(&schedule.affinity);
        for (std::size_t i = 0; i < _node["schedule"]["cpuAffinity"].size(); ++i) {
            CPU_SET(_node["schedule"]["cpuAffinity"][i].as<uint32_t>(), &schedule.affinity);
        }
    }

    schedule.valid = true;
    return 0;
}

int32_t NodeConfig::ParseTrigger() {
    DO_OR_ERROR(_node["trigger"].IsSequence(), "trigger");
    for (std::size_t k = 0; k < _node["trigger"].size(); ++k) {
        Trigger tmp;

        DO_OR_ERROR(_node["trigger"][k]["name"], "trigger.name");
        tmp.name = _node["trigger"][k]["name"].as<std::string>();

        DO_OR_ERROR(_node["trigger"][k]["type"], "trigger.type");
        tmp.type = _node["trigger"][k]["type"].as<std::string>();

        if (_node["trigger"][k]["period"]) {
            tmp.period_ms = _node["trigger"][k]["period"].as<uint32_t>();
        } else {
            tmp.period_ms = 0;
        }

        if (tmp.type == "EVENT") {
            DO_OR_ERROR(_node["trigger"][k]["mainSources"].IsSequence(), "trigger.mainSources");
            for (std::size_t i = 0; i < _node["trigger"][k]["mainSources"].size(); ++i) {
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["name"], "trigger.mainSources.name");
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["timeout"], "trigger.mainSources.timeout");
                Trigger::MainSource main_source;
                main_source.name = _node["trigger"][k]["mainSources"][i]["name"].as<std::string>();
                main_source.timeout_ms = _node["trigger"][k]["mainSources"][i]["timeout"].as<uint32_t>();

                tmp.main_sources.emplace_back(main_source);
                SetRecvInstanceAsync(main_source.name, true);
            }
        } else if (tmp.type == "PERIOD") {
            DO_OR_ERROR(tmp.period_ms != 0, "trigger.period");
        } else if (tmp.type == "TS_ALIGN") {
            DO_OR_ERROR(_node["trigger"][k]["mainSources"].IsSequence(), "trigger.mainSources");
            for (std::size_t i = 0; i < _node["trigger"][k]["mainSources"].size(); ++i) {
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["name"], "trigger.mainSources.name");
                DO_OR_ERROR(_node["trigger"][k]["mainSources"][i]["timeout"], "trigger.mainSources.timeout");
                Trigger::MainSource main_source;
                main_source.name = _node["trigger"][k]["mainSources"][i]["name"].as<std::string>();
                main_source.timeout_ms = _node["trigger"][k]["mainSources"][i]["timeout"].as<uint32_t>();

                tmp.main_sources.emplace_back(main_source);
                SetRecvInstanceAsync(main_source.name, true);
            }

            DO_OR_ERROR(_node["trigger"][k]["timeWindow"], "trigger.timeWindow");
            tmp.time_window_ms = _node["trigger"][k]["timeWindow"].as<uint32_t>();

            if (_node["trigger"][k]["alignTimeoutMs"]) {
                tmp.align_timeout_ms = _node["trigger"][k]["alignTimeoutMs"].as<uint32_t>();
            } else {
                tmp.align_timeout_ms = 0;
            }

            if (_node["trigger"][k]["alignValidityMs"]) {
                tmp.align_validity_ms = _node["trigger"][k]["alignValidityMs"].as<uint32_t>();
            } else {
                tmp.align_validity_ms = 2 * tmp.time_window_ms;
            }
        } else {
            ADF_EARLY_LOG << "Unsupported trigger type " << tmp.type;
            return -1;
        }

        if (_node["trigger"][k]["auxSources"].IsSequence()) {
            for (std::size_t i = 0; i < _node["trigger"][k]["auxSources"].size(); ++i) {
                Trigger::AuxSource aux_source;

                DO_OR_ERROR(_node["trigger"][k]["auxSources"][i]["name"], "trigger.auxSources.name");
                aux_source.name = _node["trigger"][k]["auxSources"][i]["name"].as<std::string>();

                if (_node["trigger"][k]["auxSources"][i]["multiFrame"]) {
                    aux_source.multi_frame = _node["trigger"][k]["auxSources"][i]["multiFrame"].as<uint32_t>();
                } else {
                    aux_source.multi_frame = 0;
                }

                if (_node["trigger"][k]["auxSources"][i]["readClear"]) {
                    aux_source.read_clear = _node["trigger"][k]["auxSources"][i]["readClear"].as<bool>();
                } else {
                    aux_source.read_clear = false;
                }

                tmp.aux_sources.emplace_back(aux_source);
            }
        }

        if (_node["trigger"][k]["expExecTimeMs"]) {
            tmp.exp_exec_time_ms = _node["trigger"][k]["expExecTimeMs"].as<uint64_t>();
        }

        trigger.emplace_back(tmp);
    }

    return 0;
}

int32_t NodeConfig::ParseThreadPool() {
    if (!_node["threadPool"]) {
        thread_pool.num = 0;
        return 0;
    }

    DO_OR_ERROR(_node["threadPool"]["threadNum"], "threadPool.threadNum");
    thread_pool.num = _node["threadPool"]["threadNum"].as<uint32_t>();

    return 0;
}

int32_t NodeConfig::ParseProfiler() {
    if (!_node["profiler"]) {
        profiler.enable = false;
        return 0;
    }

    DO_OR_ERROR(_node["profiler"]["name"], "profiler.name");
    profiler.name = _node["profiler"]["name"].as<std::string>();

    DO_OR_ERROR(_node["profiler"]["enable"], "profiler.enable");
    profiler.enable = _node["profiler"]["enable"].as<bool>();

    if (!_node["profiler"]["latency"]) {
        profiler.latency.enable = false;
    } else {
        DO_OR_ERROR(_node["profiler"]["latency"]["enable"], "profiler.latency.enable");
        profiler.latency.enable = _node["profiler"]["latency"]["enable"].as<bool>();

        if (_node["profiler"]["latency"]["link"]) {
            for (std::size_t i = 0; i < _node["profiler"]["latency"]["link"].size(); ++i) {
                LatencyLink link;

                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["name"], "profiler.latency.link[].name");
                link.name = _node["profiler"]["latency"]["link"][i]["name"].as<std::string>();
                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["recvMsg"], "profiler.latency.link[].recvMsg");
                link.recv_msg = _node["profiler"]["latency"]["link"][i]["recvMsg"].as<std::string>();
                DO_OR_ERROR(_node["profiler"]["latency"]["link"][i]["sendMsg"], "profiler.latency.link[].sendMsg");
                link.send_msg = _node["profiler"]["latency"]["link"][i]["sendMsg"].as<std::string>();
                profiler.latency.links.emplace_back(link);
            }
        }

        if (_node["profiler"]["latency"]["show"]) {
            for (std::size_t i = 0; i < _node["profiler"]["latency"]["show"].size(); ++i) {
                LatencyShow show;

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["link_name"], "profiler.latency.show[].link_name");
                show.link_name = _node["profiler"]["latency"]["show"][i]["link_name"].as<std::string>();

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["trigger_name"],
                            "profiler.latency.show[].trigger_name");
                show.trigger_name = _node["profiler"]["latency"]["show"][i]["trigger_name"].as<std::string>();

                DO_OR_ERROR(_node["profiler"]["latency"]["show"][i]["from_msg"], "profiler.latency.show[].from_msg");
                show.from_msg = _node["profiler"]["latency"]["show"][i]["from_msg"].as<std::string>();
                profiler.latency.shows.emplace_back(show);
            }
        }
    }

    if (!_node["profiler"]["checkpoint"]) {
        profiler.checkpoint.enable = false;
    } else {
        DO_OR_ERROR(_node["profiler"]["checkpoint"]["enable"], "profiler.checkpoint.enable");
        profiler.checkpoint.enable = _node["profiler"]["checkpoint"]["enable"].as<bool>();
    }

    return 0;
}

int32_t NodeConfig::ParseMonitor() {
    monitor.freq_enable = true;
    monitor.latency_enable = false;
    monitor.print_period_ms = 10000;
    if (!_node["monitor"]) {
        return 0;
    }
    if (_node["monitor"]["freqEnable"]) {
        monitor.freq_enable = _node["monitor"]["freqEnable"].as<bool>();
    }
    if (_node["monitor"]["latencyEnable"]) {
        monitor.latency_enable = _node["monitor"]["latencyEnable"].as<bool>();
    }

    if (_node["monitor"]["printPeriodMs"]) {
        monitor.print_period_ms = _node["monitor"]["printPeriodMs"].as<uint32_t>();
    }

    return 0;
}

void NodeConfig::SetRecvInstanceAsync(const std::string& name, bool is_async) {
    for (auto& instance : recv_instance_configs) {
        instance.is_async = is_async;
    }
}
}  // namespace adf
}  // namespace netaos
}  // namespace hozon