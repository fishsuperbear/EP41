
#include "config_loader.h"
#include <sys/types.h>

#include <cstdint>
#include <fstream>

#include "em/include/proctypes.h"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace sensor {

#define PARSE_YAML_CONFIG(config, type, key, value) \
    if(!config[key]) {          \
        return false;           \
    }                           \
    else {                          \
        value = config[key].as<type>();\
    }


uint32_t ConfigLoader::log_level_ = 2;  // default: info
uint32_t ConfigLoader::log_mode_ = 3;   // default: remote
std::string ConfigLoader::log_file_ = "/opt/usr/log/soc_log/"; // default: current path
uint32_t ConfigLoader::nnp_ = 1;

bool ConfigLoader::LoadConfig(const std::string file) {
    std::ifstream fin(file);
    if (!fin) {
        return false;
    } else {
        fin.close();
    }
    YAML::Node config = YAML::LoadFile(file);
    if(!config) {
        return false;
    }
    
    PARSE_YAML_CONFIG(config, uint32_t, "logLevel", log_level_);
    PARSE_YAML_CONFIG(config, uint32_t, "logMode", log_mode_);
    PARSE_YAML_CONFIG(config, std::string, "file", log_file_);
    PARSE_YAML_CONFIG(config, uint32_t, "nppOrHpp", nnp_);

    return true;
}

}  // namespace sensor
}   //namespace netaos
}  // namespace hozon
