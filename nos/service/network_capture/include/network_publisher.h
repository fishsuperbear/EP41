/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef NETWORK_PUBLISHER_H
#define NETWORK_PUBLISHER_H
#pragma once
#include <string>
#include <vector>
#include <map>
#include <queue>
#include "network_capture/include/someip_struct_define.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"

#include "proto/soc/raw_point_cloud.pb.h"
#include "cm/include/skeleton.h"
namespace hozon {
namespace netaos {
namespace network_capture {

static const std::string lidar_topic = "/soc/rawpointcloud";
class NetworkPub {
   public:  
    void Init();
    void Run();
    void Stop();
    void Deinit();

    void SetTopics(std::vector<std::string>& topics);
    bool Lidar_IsMatched();
    bool someip_IsMatched();
    void Set_Lidar_State(const bool& num) {
        lidar_state = num;
    }
    void Set_Someip_State(const bool& num) {
        someip_state = num;
    }

   private:
    template <typename T>
    bool Send(const std::string& topic, T& msg);
    bool SendSomeip(const std::string& topic, const std::vector<char>& msg);
    void LidarPublish();
    void SomeipPublish();

   private:
    bool stop_flag_ = false;
    bool lidar_state = false;
    bool someip_state = false;

    std::map<std::string, std::unique_ptr<hozon::netaos::cm::Skeleton>> skeletons_;

    std::unique_ptr<std::thread> lidar_pub_thread_;
    std::shared_ptr<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>> lidar_pub_list_;
    std::shared_ptr<std::mutex> lidar_mtx_;

    std::unique_ptr<std::thread> someip_pub_thread_;
    std::shared_ptr<std::queue<std::unique_ptr<raw_someip_message>>> someip_pub_list_;
    std::shared_ptr<std::mutex> someip_mtx_;

    std::unique_ptr<CmProtoBuf> cm_idl_data;
    std::unique_ptr<CmSomeipBuf> cm_idl_someip_data;

    uint32_t lidar_msg_count;
    uint32_t someip_msg_count;
    std::string serialize_str;

    std::vector<std::string> someip_topic_list_;

   public:
    NetworkPub(const std::shared_ptr<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>> &lidar_pub_list, const std::shared_ptr<std::mutex> &lidar_mtx,
               const std::shared_ptr<std::queue<std::unique_ptr<raw_someip_message>>> &someip_pub_list, const std::shared_ptr<std::mutex> &someip_mtx,
               const std::map<std::uint32_t, std::string> someip_topic_map)
    : lidar_pub_list_(lidar_pub_list)
    , lidar_mtx_(lidar_mtx) 
    , someip_pub_list_(someip_pub_list)
    , someip_mtx_(someip_mtx)
    {
        skeletons_[lidar_topic] = nullptr;
        for (const auto& topic : someip_topic_map) {
            skeletons_[topic.second] = nullptr;
            someip_topic_list_.emplace_back(topic.second);
        }
    }
    ~NetworkPub();
};

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
#endif