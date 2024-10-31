/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef YUV_NVSTREAM_RECEIVER_V2_H
#define YUV_NVSTREAM_RECEIVER_V2_H
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "sensor/nvs_consumer/CEncManager.h"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"
// #include "include/multicast.h"
#include "camera_venc_config.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

// struct YuvBufWrapper {
//     void *pre_fence = nullptr;
//     void *eof_fence = nullptr;
//     void *packet = nullptr;

//     std::function<void(void *packet, void *pref_fence, void *eof_fence)> release;

//     ~YuvBufWrapper();
// };

struct CompareById {
    bool operator()(const std::string& a, const std::string& b) const {
        auto prefix_idx_a = a.find("_");
        auto prefix_idx_b = b.find("_");
        auto x = a.substr(prefix_idx_a);
        auto y = a.substr(prefix_idx_b);
        auto id_a = std::stoi(a.substr(prefix_idx_a + 1));
        auto id_b = std::stoi(b.substr(prefix_idx_b + 1));
        return id_a < id_b;
    }
};

// TODO(mxt): replace by vector later?
using SensorInfoMap = std::map<std::string, SensorInfo, CompareById>;

class YuvNvStreamReceiverV2 {
   public:
    YuvNvStreamReceiverV2();
    ~YuvNvStreamReceiverV2();

    void SetTopics(std::vector<std::string>& topics);
    void SetSensorInfos(const SensorInfoMap& sensor_infos);
    int Init();
    void Deinit();

    void SetCallbacks(std::string topic, hozon::netaos::desay::EncConsumerCbs cbs);
    std::shared_ptr<struct YuvBufWrapper> Get();
    std::shared_ptr<struct YuvBufWrapper> Get(std::string& topic);

   private:
    // struct CamInfo {
    //     int sensor_id;
    //     int image_width;
    //     int image_height;

    //     CamInfo()
    //     : sensor_id(0)
    //     , image_width(0)
    //     , image_height(0) {

    //     }

    //     CamInfo(uint32_t id, uint32_t width, uint32_t height)
    //     : sensor_id(id)
    //     , image_width(width)
    //     , image_height(height) {

    //     }
    // };

    std::map<std::string, std::shared_ptr<hozon::netaos::desay::CIpcConsumerChannel>> proxys_;
    SensorInfoMap sensor_info_mapping_;
    std::map<std::string, hozon::netaos::desay::EncConsumerCbs> cbs_map_;
};

}  // namespace cameravenc
}  // namespace netaos
}  // namespace hozon
#endif
