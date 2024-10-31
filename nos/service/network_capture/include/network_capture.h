/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef NETWORK_CAPTURE_H
#define NETWORK_CAPTURE_H
#pragma once

#include "proto/soc/raw_point_cloud.pb.h"
#include "network_capture/include/network_publisher.h"
#include "network_capture/include/someip_capture_config.h"
#include "network_capture/include/someip_capture.h"
#include "network_capture/include/lidar_capture_config.h"
#include "network_capture/include/lidar_capture.h"
namespace hozon {
namespace netaos {
namespace network_capture {
#ifdef BUILD_FOR_ORIN
#define CONFIG_PATH "/app/runtime_service/network_capture/conf"
// #define CONFIG_PATH "/opt/usr/upgrade/zyj/nos_orin/runtime_service/network_capture/conf"
#else
#define CONFIG_PATH "/app/runtime_service/network_capture/conf"
// #define CONFIG_PATH "/mnt/36a47b35-4e1d-4ba4-bdae-604281ca8ca9/nos2/output/nos_x86_2004/runtime_service/network_capture/conf"
#endif
class NetworkCapture {
   public:
    NetworkCapture() = default;
    ~NetworkCapture() = default;

    std::int32_t Init();
    std::int32_t DeInit();
    std::int32_t Run();
    std::int32_t Stop();

   private:
    bool stop_flag_ = false;
    bool someip_flag_ = false;
    bool lidar_cloudpoint_flag_ = false;

   // publisher
    std::unique_ptr<NetworkPub> network_pub_;

   // lidar
    std::unique_ptr<LidarCapture> lidar_capture_;
    std::unique_ptr<LidarFilterInfo> lidar_capture_config_;
    std::shared_ptr<std::mutex> lidar_mtx_;
    std::unique_ptr<std::thread> lidar_run_thread_;
    std::shared_ptr<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>> lidar_pub_list_;

   // someip
    std::vector<std::unique_ptr<SomeipCapture>> someip_capture_list_;
    std::unique_ptr<std::vector<std::unique_ptr<SomeipFilterInfo>>> someip_capture_config_list_;
    std::shared_ptr<std::mutex> someip_mtx_;
    std::unique_ptr<std::thread> someip_run_thread_;
    std::shared_ptr<std::queue<std::unique_ptr<raw_someip_message>>> someip_pub_list_;
};

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif