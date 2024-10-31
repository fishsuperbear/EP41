#ifndef _HAL_LIDAR_HPP_
#define _HAL_LIDAR_HPP_

#include <functional>
#include <iostream>
#include <vector>

namespace hal {
namespace lidar {

struct PointsCallbackData;

typedef std::function<void(const PointsCallbackData &)> PointsCallback;

struct ConfigInfo {
    int index;
    std::string model;  // 激光雷达的型号，目前支持：hesai_at128
    std::string ip;     // lidar本身的ip地址
    int port;           // 端口号
};

struct PointXYZIT {
    float x;
    float y;
    float z;
    uint32_t intensity;
    uint64_t timestamp;  // ns
    uint8_t ring;        // 行，1-128
    uint16_t column;     // 列（暂不支持，为空）
    uint8_t confidence;  // 置信度
};

struct PointsCallbackData {
    int index;
    std::string model;
    uint64_t sequence;
    uint64_t timestamp;  // ns
    std::vector<PointXYZIT> points;
};

enum ErrorCode {
    SUCCESS = 0,
    UNKNOWN_LIDAR_MODEL,
};

/**
 * @brief start send data
 *
 * @param configs
 * @param callback
 * @return ErrorCode
 */
ErrorCode Start(const std::vector<ConfigInfo> &configs, PointsCallback callback);

/**
 * @brief stop send data
 *
 */
void Stop();

}  // namespace lidar
}  // namespace hal

#endif  // _HAL_LIDAR_HPP_
