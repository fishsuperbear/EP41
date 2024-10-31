#pragma once

#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class DSConfig {
public:
    struct DataSource {
        std::string topic;
        uint32_t capacity;
        std::string type;
        uint32_t cm_domain_id;
        std::string cm_topic;
    };

    struct Log {
        uint32_t level;
    };

    int32_t Parse(const std::string& file);

    std::vector<DataSource> data_sources_in;
    std::vector<DataSource> data_sources_out;
    Log log;
    
private:
    int32_t ParseDataSource();
    int32_t ParseLog();
    
    YAML::Node _node;
    std::string _file;
};

}
}
}