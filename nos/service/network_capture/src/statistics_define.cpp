#include "network_capture/include/statistics_define.h"
#include "network_capture/include/network_logger.h"
#include <mutex>
#include <iostream>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace network_capture {

std::uint64_t unprocessed_lidar_frame = 0;
std::uint64_t processed_lidar_frame = 0;
std::uint64_t send_lidar_frame = 0;
std::uint64_t send_someip_frame = 0;
std::atomic<std::uint64_t> unprocessed_someip_frame = 0;
std::atomic<std::uint64_t> udp_frame = 0;
std::atomic<std::uint64_t> tcp_frame = 0;
std::atomic<std::uint64_t> processed_someip_frame = 0;
std::atomic<std::uint64_t> someip_frame = 0;
// static std::atomic<std::uint64_t> processed_someip_frame = 0;
std::atomic<std::uint64_t> someip_tp_frame = 0;
std::unordered_map<std::string, std::uint64_t> count_map_;

static std::mutex inst_mutex_;
Counter* Counter::instance_ = nullptr;

Counter::Counter() {
    count_map_.clear();
}

Counter& Counter::Instance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(inst_mutex_);
        if (nullptr == instance_) {
            instance_ = new Counter();
        }
    }
    return *instance_;
}

// 原子增加计数
void Counter::increment(const std::string&& name) {
    count_map_[name]++;
}

void Counter::increment(std::atomic<std::uint64_t>& count) {
    count.fetch_add(1, std::memory_order_relaxed);
}

// 原子减少计数
void Counter::decrement(const std::string&& name) {
    count_map_[name]--;
}

void Counter::decrement(std::atomic<std::uint64_t>& count) {
    count.fetch_sub(1, std::memory_order_relaxed);
}

// 原子获取当前计数
const int Counter::getCount(const std::string&& name) {
    return count_map_[name];
}

std::uint64_t Counter::getCount(std::atomic<std::uint64_t>& count) {
    return count.load(std::memory_order_relaxed);
}

// 统计清零
void Counter::setZero() {
    for (auto& i : count_map_) {
        i.second = 0;
    }  
}

void Counter::setZero(std::atomic<std::uint64_t>& count) {
    count.store(0, std::memory_order_relaxed);
}

// const std::unordered_map<std::string, std::atomic<int>>& Counter::getmap() {
//     return count_map_;
// }

// 打印统计信息
void Counter::printStat(int64_t gap) {
    std::uint64_t count = 0;
    count = unprocessed_lidar_frame;
    NETWORK_LOG_INFO << "unprocessed_lidar_frame is : " << count << ", unprocessed_lidar_frame ratio : " << count * 1. / gap << "hz";
    count = processed_lidar_frame;
    NETWORK_LOG_INFO << "processed_lidar_frame is : " << count << ", processed_lidar_frame ratio : " << count * 1. / gap << "hz";
    count = send_lidar_frame;
    NETWORK_LOG_INFO << "send_lidar_frame is : " << count << ", send_lidar_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(unprocessed_someip_frame);
    NETWORK_LOG_INFO << "unprocessed_someip_frame is : " << count << ", unprocessed_someip_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(udp_frame);
    NETWORK_LOG_INFO << "udp_frame is : " << count << ", udp_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(tcp_frame);
    NETWORK_LOG_INFO << "tcp_frame is : " << count << ", tcp_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(processed_someip_frame);
    NETWORK_LOG_INFO << "processed_someip_frame is : " << count << ", processed_someip_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(someip_frame);
    NETWORK_LOG_INFO << "someip_frame is : " << count << ", someip_frame ratio : " << count * 1. / gap << "hz";
    count = getCount(someip_tp_frame);
    NETWORK_LOG_INFO << "someip_tp_frame is : " << count << ", someip_tp_frame ratio : " << count * 1. / gap << "hz";
    count = send_someip_frame;
    NETWORK_LOG_INFO << "send_someip_frame is : " << count << ", send_someip_frame ratio : " << count * 1. / gap << "hz";
    for (const auto& member : count_map_) {
        count = member.second;
        NETWORK_LOG_INFO << "send " << member.first << " num is : " << count << ", ratio is : " << count * 1. / gap << "hz";;
    }
    setZero();
    unprocessed_lidar_frame = 0;
    processed_lidar_frame = 0;
    send_lidar_frame = 0;
    send_someip_frame = 0;
    setZero(unprocessed_someip_frame);
    setZero(udp_frame);
    setZero(tcp_frame);
    setZero(processed_someip_frame);
    setZero(someip_frame);
    setZero(someip_tp_frame);
}

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
