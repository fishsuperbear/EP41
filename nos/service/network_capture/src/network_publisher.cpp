/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "network_capture/include/network_publisher.h"
#include "network_capture/include/network_logger.h"
#include "network_capture/include/statistics_define.h"
// /*debug*/#include "network_capture/include/function_statistics.h"
namespace hozon {
namespace netaos {
namespace network_capture {

extern std::uint64_t send_lidar_frame;
extern std::uint64_t send_someip_frame;

NetworkPub::~NetworkPub() {}

void NetworkPub::Init() {
    serialize_str.reserve(700000);
    cm_idl_data = std::make_unique<CmProtoBuf>();
    cm_idl_someip_data = std::make_unique<CmSomeipBuf>();
    for (auto& it : skeletons_) {
        if (!it.second) {
            if (it.first == lidar_topic)
                it.second = std::make_unique<hozon::netaos::cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
            else 
                it.second = std::make_unique<hozon::netaos::cm::Skeleton>(std::make_shared<CmSomeipBufPubSubType>());
            if (it.second->Init(0, it.first) < 0) {
                NETWORK_LOG_ERROR << "Init cm proxy failed. Topic: " << it.first;
                it.second.reset();
            }
        }
    }
}

void NetworkPub::Run() {
    lidar_msg_count = 0;
    someip_msg_count = 0;
    stop_flag_ = false;
    lidar_pub_thread_ = std::make_unique<std::thread>(std::thread(&NetworkPub::LidarPublish, this));
    someip_pub_thread_ = std::make_unique<std::thread>(std::thread(&NetworkPub::SomeipPublish, this));
}

void NetworkPub::Stop() {
    stop_flag_ = true;

    if (lidar_pub_thread_->joinable())
        lidar_pub_thread_->join();

    if (someip_pub_thread_->joinable())
        someip_pub_thread_->join();

    for (auto& it : skeletons_) {
        if (!it.second) {
            it.second->Deinit();
            it.second.reset();
        }
    }
}

void NetworkPub::Deinit() {}

template <typename T>
bool NetworkPub::Send(const std::string& topic, T& msg) {
    if (!skeletons_[topic]) {
        NETWORK_LOG_WARN << "Cm proxy is not inited. Topic: " << topic;
        return false;
    }

    if (!skeletons_[topic]->IsMatched()) {
        NETWORK_LOG_WARN << "Cm proxy is not matched yet. Topic: " << topic;
        return false;
    }

    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);
    double pub_time = static_cast<double>(time_now.tv_sec) + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
    msg->mutable_header()->set_publish_stamp(pub_time);

    if (nullptr == cm_idl_data)
        cm_idl_data = std::make_unique<CmProtoBuf>();
    cm_idl_data->name(msg->GetTypeName());
    serialize_str.clear();
    if (!msg->SerializeToString(&serialize_str)) {
        NETWORK_LOG_ERROR << "Serialize network_capture message to string failed. Topic: " << topic;
        return false;
    }
    // NETWORK_LOG_DEBUG << "serialize_str.size() : " << serialize_str.size();
    cm_idl_data->str(std::vector<char>(std::make_move_iterator(serialize_str.begin()), std::make_move_iterator(serialize_str.end())));
    if (skeletons_[topic]->Write(std::move(cm_idl_data)) < 0) {
        NETWORK_LOG_WARN << "Write data to cm failed. Topic: " << topic;
        return false;
    }
    
    send_lidar_frame++;
    return true;
}

bool NetworkPub::SendSomeip(const std::string& topic, const std::vector<char>& msg) {
    if (!skeletons_[topic]) {
        NETWORK_LOG_WARN << "Cm proxy is not inited. Topic: " << topic;
        return false;
    }

    if (!skeletons_[topic]->IsMatched()) {
        NETWORK_LOG_WARN << "Cm proxy is not matched yet. Topic: " << topic;
        return false;
    }

    if (nullptr == cm_idl_someip_data)
        cm_idl_someip_data = std::make_unique<CmSomeipBuf>();
    cm_idl_someip_data->name(topic);
    cm_idl_someip_data->str(msg);

    if (skeletons_[topic]->Write(std::move(cm_idl_someip_data)) < 0) {
        NETWORK_LOG_WARN << "Write data to cm failed. Topic: " << topic;
        return false;
    }
    
    send_someip_frame++;
    return true;
}

void NetworkPub::LidarPublish() {
    bool empty = false;
    while (!stop_flag_) {
        if (!lidar_state) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(*lidar_mtx_);
            // NETWORK_LOG_INFO << "lidar_pub_list_.size(network_pub_->) : " << lidar_pub_list_->size();
            empty = lidar_pub_list_->empty();
        }
        if (!empty) {
            std::lock_guard<std::mutex> lk(*lidar_mtx_);
            Send(lidar_topic, lidar_pub_list_->front());
            lidar_pub_list_->pop();
            std::cout << "Sending " << ++lidar_msg_count << " RawPointCloud messages, Sending " << someip_msg_count << " Someip messages...\r";
            std::cout.flush();
        } else if (!stop_flag_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    while (lidar_pub_list_->size() > 0)
    {
        lidar_pub_list_->pop();
    }
    
}

void NetworkPub::SomeipPublish() {
    bool empty = false;
    while (!stop_flag_) {
        if (!someip_state) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(*someip_mtx_);
            empty = someip_pub_list_->empty();
        }
        if (!empty) {
            std::lock_guard<std::mutex> lk(*someip_mtx_);
            SendSomeip(someip_pub_list_->front()->topic, someip_pub_list_->front()->msg);
            someip_pub_list_->pop();
            std::cout << "Sending " << lidar_msg_count << " RawPointCloud messages, Sending " << ++someip_msg_count << " Someip messages...\r";
            std::cout.flush();
        } else if (!stop_flag_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    while (someip_pub_list_->size() > 0)
    {
        someip_pub_list_->pop();
    }
}

void NetworkPub::SetTopics(std::vector<std::string>& topics) {
    for (auto& topic : topics) {
        skeletons_[topic] = nullptr;
    }
}

bool NetworkPub::Lidar_IsMatched() {
    return skeletons_[lidar_topic]->IsMatched();
}

bool NetworkPub::someip_IsMatched() {
    bool ret = false;
    for (const auto& topic : someip_topic_list_) {
        ret = ret | skeletons_[topic]->IsMatched();
    }
    return ret;
}

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon