/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef LIDAR_CAPTURE_H
#define LIDAR_CAPTURE_H
#pragma once
#include "proto/dead_reckoning/dr.pb.h"
#include "proto/soc/raw_point_cloud.pb.h"
#include "network_capture/include/base_capture.h"
#include "network_capture/include/lidar_capture_config.h"
#include "network_capture/include/lidar_struct_define.h"
#include "network_capture/include/network_logger.h"
#include "cm/include/proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
namespace hozon {
namespace netaos {
namespace network_capture {

class LidarCapture : protected BaseCapture {
   public:
    std::int32_t Init() override;
    std::int32_t DeInit() override;
    std::int32_t Run() override;
    std::int32_t Stop() override;

   protected:
    void capPacket(std::string eth_name, std::string filter_exp) override;
    void process_packet(const u_char *packet, pcap_pkthdr& header) override;

   private:
    // void Parse(uint8_t* dataptr, uint32_t size);
    void dead_reckoning_receiver();
    void frame_ratio_info();

   private:
    bool stop_flag_ = false;
    std::string filter_exp;
    LidarFilterInfo lidar_filter_info_;
    std::unique_ptr<std::thread> lidar_thread_;
    std::unique_ptr<std::thread> receive_thread_;
    // std::unique_ptr<std::thread> frame_ratio_info_thread_;

    std::unique_ptr<hozon::soc::RawPointCloud> send_data_ptr_;
    std::shared_ptr<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>> lidar_pub_list_;
    std::shared_ptr<std::mutex> mtx_;
    std::vector<SubPointCloudFrameDEA> frame_list_;

    std::unique_ptr<hozon::netaos::cm::Proxy> dead_reckoning_proxy_;
    std::shared_ptr<CmProtoBuf> latest_dead_reckoning_data_;
    std::mutex proxy_mtx_;

    bool framing_flag = false;
    bool start_frame_flag = false;
    int32_t pack_num = 0;
    int32_t Azimuth_last;
    float fov_angle = 0;
    std::time_t timestamp;
    std::tm tm_time = {0};
    uint64_t seq_ = 0;
    // uint64_t frame_count = 0;

   public:
    LidarCapture(){}
    LidarCapture(const LidarFilterInfo &info, const std::shared_ptr<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>> &list, const std::shared_ptr<std::mutex> &mtx) 
    : lidar_filter_info_(info)
    , lidar_pub_list_(list)
    , mtx_(mtx) {}
};

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif