#include "can_tsync_center/can_tsync_config_parser.h"
#include "can_tsync_center/can_tsync_logger.h"

namespace hozon {
namespace netaos {
namespace tsync {

#define DO(statement) \
    if ((statement) < 0) { \
        return -1; \
    }

#define DO_OR_ERROR(statement, str) \
    if (!(statement)) { \
        CTSC_LOG_ERROR << "Fail to parse " << (str) << " in config file."; \
        return -1; \
    }

int32_t ConfigParser::ParseCanTsyncConfig(const std::string& file, std::vector<CanTSyncConfig>& config) {
    YAML::Node node = YAML::LoadFile(file);
    if (!node) {
        CTSC_LOG_INFO << "Fail to load config file " << file;
        return -1;
    }

    DO_OR_ERROR(node["interfaces"].IsSequence(), "interfaces")
    config.resize(node["interfaces"].size());
    for (std::size_t i = 0; i < node["interfaces"].size(); ++i) {
        DO_OR_ERROR(node["interfaces"][i]["name"], "interfaces.name")
        config[i].interface = node["interfaces"][i]["name"].as<std::string>();

        DO_OR_ERROR(node["interfaces"][i]["type"], "interfaces.type")
        std::string type = node["interfaces"][i]["type"].as<std::string>();
        if (type == "CANFD_16_BYTE") {
            config[i].type = TSyncFrameType::CANFD_16_BYTE;
        }
        else if (type == "CANFD_8_BYTE") {
            config[i].type = TSyncFrameType::CANFD_8_BYTE;
        }
        else if (type == "CAN_STANDARD") {
            config[i].type = TSyncFrameType::CAN_STANDARD;
        }
        else {
            CTSC_LOG_ERROR << "Unsupported type " << type;
            return -1;
        }

        DO_OR_ERROR(node["interfaces"][i]["can_id"], "interfaces.can_id")
        config[i].can_id = node["interfaces"][i]["can_id"].as<uint32_t>();

        DO_OR_ERROR(node["interfaces"][i]["crc_enable"], "interfaces.crc_enable")
        config[i].crc_enable = node["interfaces"][i]["crc_enable"].as<bool>();

        DO_OR_ERROR(node["interfaces"][i]["interval_ms"], "interfaces.interval_ms")
        config[i].interval_ms = node["interfaces"][i]["interval_ms"].as<uint32_t>();

        DO_OR_ERROR(node["interfaces"][i]["timeout_ms"], "interfaces.timeout_ms")
        config[i].timeout_ms = node["interfaces"][i]["timeout_ms"].as<uint32_t>();

        if (config[i].crc_enable) {
            DO_OR_ERROR(node["interfaces"][i]["data_id"].IsSequence(), "interfaces.data_id")
            DO_OR_ERROR(node["interfaces"][i]["data_id"].size() == 16, "size of interfaces.data_id")
            for (std::size_t j = 0; j < node["interfaces"][i]["data_id"].size(); ++j) {
                config[i].data_id[j] = node["interfaces"][i]["data_id"][j].as<uint32_t>();
            }
        }
    }

    return 0;
}

int32_t ConfigParser::ParseLogConfig(const std::string& file, LogConfig& config) {
    YAML::Node node = YAML::LoadFile(file);
    if (!node) {
        CTSC_LOG_INFO << "Fail to load config file " << file;
        return -1;
    }

    DO_OR_ERROR(node["log"], "log")

    DO_OR_ERROR(node["log"]["level"], "log.level")
    config.level = node["log"]["level"].as<uint32_t>();

    DO_OR_ERROR(node["log"]["mode"], "log.mode")
    config.mode = node["log"]["mode"].as<uint32_t>();

    DO_OR_ERROR(node["log"]["file"], "log.file")
    config.file = node["log"]["file"].as<std::string>();

    return 0;
}

}
}
}