#include "camera_venc_config.h"
#include "yaml-cpp/yaml.h"
#include "camera_venc_logger.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

template <typename T>
void GetMemberValue(T& var, std::string member_name, YAML::Node node) {
    var = node[member_name].as<T>();
}

#define CHECK_MEMBER(member_name, node, ret) \
    if (!node[member_name].IsDefined()) { \
        CAMV_CRITICAL << "Camera venc config " << member_name << " does not exist."; \
        return ret; \
    }

#define GET_MEMBER(var, member_name, node, ret) \
    CHECK_MEMBER(member_name, node,ret) \
    GetMemberValue(var, member_name, node);

std::shared_ptr<CameraVencConfig> CameraVencConfig::LoadConfig(std::string file_path) {

    YAML::Node config_node = YAML::LoadFile(file_path);

    CameraVencConfig cfg;
    GET_MEMBER(cfg.sensor_ids, "sensor_ids", config_node, nullptr);
    GET_MEMBER(cfg.codec_type, "codec_type", config_node, nullptr);
    GET_MEMBER(cfg.write_file, "write_file", config_node, nullptr);
    GET_MEMBER(cfg.uhp_mode, "uhp_mode", config_node, nullptr);
    GET_MEMBER(cfg.frame_sampling, "frame_sampling", config_node, nullptr);

    CHECK_MEMBER("sensor_infos", config_node, nullptr);
    YAML::Node sensor_infos_node = config_node["sensor_infos"];
    for (auto it = sensor_infos_node.begin(); it != sensor_infos_node.end(); ++it) {

        SensorInfo si;
        YAML::Node sensor_info_node = *it;
        GET_MEMBER(si.sensor_id, "sensor_id", sensor_info_node, nullptr);
        GET_MEMBER(si.resolution_width, "resolution_width", sensor_info_node, nullptr);
        GET_MEMBER(si.resolution_height, "resolution_height", sensor_info_node, nullptr);
        GET_MEMBER(si.src_layout, "src_layout", sensor_info_node, nullptr);
        GET_MEMBER(si.yuv_topic, "yuv_topic", sensor_info_node, nullptr);
        GET_MEMBER(si.enc_topic, "enc_topic", sensor_info_node, nullptr);
        cfg.sensor_infos.push_back(si);
    }

    auto cfg_ptr = std::make_shared<CameraVencConfig>(std::move(cfg));

    return cfg_ptr;
}

}
}
}