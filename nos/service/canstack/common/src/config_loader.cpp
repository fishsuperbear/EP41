/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: 通讯组件（CM）event同步调用 示例程序客户端源码文件
 */

#include "config_loader.h"

#include <fstream>

// #include "canstack_e2e.h"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace canstack {

#define PARSE_YAML_CONFIG(config, type, key, value) \
    if(!config[key]) {          \
        return false;           \
    }                           \
    else {                          \
        value = config[key].as<type>();\
    }

bool ConfigLoader::debug_on_ = true;
bool ConfigLoader::analysys_on_ = false;
bool ConfigLoader::select_on_ = false;
std::vector<std::string> ConfigLoader::can_port_;
#ifdef CHASSIS_DEBUG_ON
bool ConfigLoader::version_for_EP40_ = false;
bool ConfigLoader::time_diff_ = false;
bool ConfigLoader::isMock_ = false;
#endif
#ifdef PROCESS_NAME_SUPPORT
std::vector<std::string> ConfigLoader::process_name_;
#endif
std::vector<std::string> ConfigLoader::log_app_name_;
uint32_t ConfigLoader::log_level_ = 2;  // default: info
uint32_t ConfigLoader::log_mode_ = 3;   // default: remote
std::string ConfigLoader::log_file_ = "./"; // default: current path
// bool ConfigLoader::version_T79_E2E_ = false;

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
    PARSE_YAML_CONFIG(config, bool, "debugOn", debug_on_);
    PARSE_YAML_CONFIG(config, bool, "analysysOn", analysys_on_);
    PARSE_YAML_CONFIG(config, bool, "selectOn", select_on_);
    PARSE_YAML_CONFIG(config, std::vector<std::string>, "canPort", can_port_);
// #ifdef CHASSIS_DEBUG_ON
//     PARSE_YAML_CONFIG(config, bool, "version_for_EP40_", version_for_EP40_);
//     PARSE_YAML_CONFIG(config, bool, "time_diff_", time_diff_);
//     PARSE_YAML_CONFIG(config, bool, "isMock", isMock_);
// #endif
// #ifdef PROCESS_NAME_SUPPORT
//     PARSE_YAML_CONFIG(config, std::vector<std::string>, "processName", process_name_);
// #endif
    PARSE_YAML_CONFIG(config, std::vector<std::string>, "logAppName", log_app_name_);
    PARSE_YAML_CONFIG(config, uint32_t, "logLevel", log_level_);
    PARSE_YAML_CONFIG(config, uint32_t, "logMode", log_mode_);
    PARSE_YAML_CONFIG(config, std::string, "file", log_file_);
// #ifdef T79_E2E_ON
//     PARSE_YAML_CONFIG(config, bool, "version_T79_E2E", version_T79_E2E_);
// #endif
    // TODO: Add E2E config
    // E2ESupervision::Instance()->Init(config);

    return true;
}

}  // namespace canstack
}
}  // namespace hozon
