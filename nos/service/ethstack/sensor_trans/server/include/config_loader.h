#pragma once 

#include <cstdint>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace sensor {

class ConfigLoader {
public:
    static bool LoadConfig(const std::string file);
    static uint32_t log_level_;
    static uint32_t log_mode_;
    static std::string log_file_;
    static uint32_t nnp_;
};

}  // namespace sensor
}   // namespace netaos
}  // namespace hozon

