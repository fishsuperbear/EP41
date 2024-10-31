/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: can subscriber abstract class
 */

#ifndef CANSTACK_CONFIG_LOADER_H
#define CANSTACK_CONFIG_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace canstack {

class ConfigLoader {
   public:
    static bool LoadConfig(const std::string file);

    static bool debug_on_;
    static bool analysys_on_;
    static bool select_on_;
    static std::vector<std::string> can_port_;
#ifdef CHASSIS_DEBUG_ON
    static bool version_for_EP40_;
    static bool time_diff_;
#endif
#ifdef PROCESS_NAME_SUPPORT
    static std::vector<std::string> process_name_;
#endif
    static std::vector<std::string> log_app_name_;
    static uint32_t log_level_;
    static uint32_t log_mode_;
    static std::string log_file_;
    // static bool version_T79_E2E_;
    static bool isMock_;
};

}  // namespace canstack
}
}  // namespace hozon

#endif  // CANSTACK_CONFIG_LOADER_H
