#pragma once

#include <cstdint>
#include <string>
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace tsync {

enum class TSyncFrameType {
    CANFD_16_BYTE,
    CANFD_8_BYTE,
    CAN_STANDARD
};

struct CanTSyncConfig {
    std::string interface;
    TSyncFrameType type;
    uint16_t can_id;
    bool crc_enable;
    uint32_t interval_ms;
    uint32_t timeout_ms;
    uint8_t data_id[16];
};

struct LogConfig {
    uint32_t level;
    uint32_t mode;
    std::string file;
};

class ConfigParser {
public:
    static int32_t ParseCanTsyncConfig(const std::string& file, std::vector<CanTSyncConfig>& config);
    static int32_t ParseLogConfig(const std::string& file, LogConfig& config);
};

}
}
}