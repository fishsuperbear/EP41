#include "adf-lite/ds/ds_config.h"
#include "adf-lite/ds/ds_logger.h"

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
        DS_LOG_ERROR << "Cannot find " << (str) << " in config file " << file; \
        return -1; \
    }

int32_t DSConfig::Parse(const std::string& file) {
    _file = file;

    _node = YAML::LoadFile(file);
    if (!_node) {
        DS_LOG_ERROR << "Fail to load config file " << file;
        return -1;
    }

    DO(ParseDataSource());
    DO(ParseLog())

    return 0;
}

int32_t DSConfig::ParseLog() {
    DO_OR_ERROR(_node["log"], "log", _file);
    DO_OR_ERROR(_node["log"]["level"], "log.level", _file);

    log.level = _node["log"]["level"].as<uint32_t>();
    return 0;
}

int32_t DSConfig::ParseDataSource() {
    if (_node["dataSourcesIn"]) {
        for (size_t i = 0; i < _node["dataSourcesIn"].size(); ++i) {
            DO_OR_ERROR(_node["dataSourcesIn"][i]["topic"], "dataSourcesIn.topic", _file);
            DO_OR_ERROR(_node["dataSourcesIn"][i]["type"], "dataSourcesIn.type", _file);
            DO_OR_ERROR(_node["dataSourcesIn"][i]["cmDomainId"], "dataSourcesIn.cmDomainId", _file);
            DO_OR_ERROR(_node["dataSourcesIn"][i]["cmTopic"], "dataSourcesIn.cmTopic", _file);

            DataSource data_source;
            data_source.topic = _node["dataSourcesIn"][i]["topic"].as<std::string>();
            data_source.type = _node["dataSourcesIn"][i]["type"].as<std::string>();
            data_source.cm_domain_id = _node["dataSourcesIn"][i]["cmDomainId"].as<uint32_t>();
            data_source.cm_topic = _node["dataSourcesIn"][i]["cmTopic"].as<std::string>();
            data_sources_in.emplace_back(data_source);
        }
    }

    if (_node["dataSourcesOut"]) {
        for (size_t i = 0; i < _node["dataSourcesOut"].size(); ++i) {
            DO_OR_ERROR(_node["dataSourcesOut"][i]["topic"], "dataSourcesOut.topic", _file);
            DO_OR_ERROR(_node["dataSourcesOut"][i]["capacity"], "dataSourcesOut.capacity", _file);
            DO_OR_ERROR(_node["dataSourcesOut"][i]["type"], "dataSourcesOut.type", _file);
            DO_OR_ERROR(_node["dataSourcesOut"][i]["cmDomainId"], "dataSourcesOut.cmDomainId", _file);
            DO_OR_ERROR(_node["dataSourcesOut"][i]["cmTopic"], "dataSourcesOut.cmTopic", _file);

            DataSource data_source;
            data_source.topic = _node["dataSourcesOut"][i]["topic"].as<std::string>();
            data_source.capacity = _node["dataSourcesOut"][i]["capacity"].as<std::uint32_t>();
            data_source.type = _node["dataSourcesOut"][i]["type"].as<std::string>();
            data_source.cm_domain_id = _node["dataSourcesOut"][i]["cmDomainId"].as<uint32_t>();
            data_source.cm_topic = _node["dataSourcesOut"][i]["cmTopic"].as<std::string>();
            data_sources_out.emplace_back(data_source);
        }
    }

    return 0;
}

}
}
}