#pragma once
#include <iostream>
#include "yaml-cpp/yaml.h"
#include "dl_adf/include/log.h"

namespace hozon {
namespace netaos {
namespace dl_adf {
#define DO_OR_ERROR(statement, str) \
    if(!(statement)) {              \
        DL_EARLY_LOG << "Cannot find " << (str) << " in config file."; \
        return -1;    \
    }                   

class NodeConfig {
public:
    NodeConfig() = default;
    ~NodeConfig() = default;
    int32_t Parse(const std::string& file);
    struct Module {
        std::string class_name;
        std::string path;
    };
    Module module;
    
private:
    YAML::Node _node;
    
};
}   // namespace dl_adf
}   // namespace netaos
}   // namespace hozon