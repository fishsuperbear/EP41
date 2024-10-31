/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CAMERA_VENC_CONFIG_H
#define CAMERA_VENC_CONFIG_H
#pragma once

#include <memory>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace cameravenc {

struct SensorInfo {
    uint32_t sensor_id;
    uint32_t resolution_width;
    uint32_t resolution_height;
    uint32_t src_layout;    // 0: BL, 1: PL
    std::string yuv_topic;
    std::string enc_topic;
};

// YCS_ADD_STRUCT(SensorInfo, sensor_id, resolution_width, resolution_height, yuv_topic, enc_topic)

struct CameraVencConfig {
    std::vector<uint32_t> sensor_ids;
    uint32_t codec_type;
    bool write_file;
    bool uhp_mode;
    uint32_t frame_sampling;
    std::vector<SensorInfo> sensor_infos;

    static std::shared_ptr<CameraVencConfig> LoadConfig(std::string file_path);
    CameraVencConfig()
    : codec_type(0)
    , write_file(false)
    , uhp_mode(false)
    , frame_sampling(0) {

    }

    CameraVencConfig(CameraVencConfig& other)
    : sensor_ids(other.sensor_ids)
    , codec_type(other.codec_type)
    , write_file(other.write_file)
    , uhp_mode(other.uhp_mode)
    , frame_sampling(other.frame_sampling)
    , sensor_infos(other.sensor_infos) {

    }

    CameraVencConfig(CameraVencConfig&& other)
    : sensor_ids(std::move(other.sensor_ids))
    , codec_type(other.codec_type)
    , write_file(other.write_file)
    , uhp_mode(other.uhp_mode)
    , frame_sampling(other.frame_sampling)
    , sensor_infos(std::move(other.sensor_infos)) {
        
    }
};

// YCS_ADD_STRUCT(CameraVencConfig, sensor_ids, codec_type, sensor_infos)

}
}
}

#endif