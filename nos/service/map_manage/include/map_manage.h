#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "proto/map/avp_map_origin.pb.h"

namespace hozon {
namespace netaos {

class MapManage {
   public:
    struct MapContent {
        uint32_t id = 0;  // 0为默认值，表示空id
        perception_map::semanticmap::AvpTaskInfo path;
        perception_map::semanticmap::AVPMap map;
    };

    struct Map {
        MapContent map_planning;
        MapContent map_slam;
        std::string feature_map;
    };

    static MapManage& getInstance(const std::string map_position = "/opt/usr/hz_map/ntp/") {
        static MapManage instance(map_position);
        return instance;
    }

    MapManage(const MapManage&) = delete;
    MapManage& operator=(const MapManage&) = delete;

    // 拉取本地所有地图ID列表，size表示地图数量
    // 在新增、更新、删除后需重新获取
    std::vector<uint32_t> pollAllMap();

    // 根据ID查询经纬度,
    // 返回 {longitude,latitude}
    std::pair<double, double> getXY(uint32_t id);

    // 删除地图
    // 返回状态，-1 表示失败，0 表示成功
    int deleteMap(uint32_t id);

    // 保存地图
    // 返回状态，-1 表示失败，>0 表示保存成功，并返回有效id（1~99）
    int saveMap(const Map& map);

    // 更新地图
    // 返回状态，-1 表示失败，0 表示成功
    int updateMap(uint32_t id, const Map& map);

    // 设置Map Id
    // 返回状态，-1 表示失败，0 表示成功
    // 未获取到符合条件的地图id写0
    int setMapId(uint32_t id);

    // 获取Map Id
    // 返回状态，0 表示默认无效id，>0 表示有效id（1~99）
    uint32_t getMapId();

    // 读取MapPlanning
    // 返回状态，nullptr表示失败
    std::shared_ptr<MapContent> getMapPlanning(uint32_t id);
    std::shared_ptr<MapContent> getMapPlanning();

    // 读取MapSlam
    // 返回状态，nullptr表示失败
    std::shared_ptr<MapContent> getMapSlam(uint32_t id);
    std::shared_ptr<MapContent> getMapSlam();

    // 读取FeatureMap
    // 返回状态，nullptr表示失败
    std::shared_ptr<std::string> getFeatureMap(uint32_t id);
    std::shared_ptr<std::string> getFeatureMap();

    // 将角度转换为弧度
    double toRadians(double degrees) { return degrees * M_PI / 180.0; }

    // 根据经纬度坐标计算距离
    double calculateDistance(double lat1, double lon1, double lat2, double lon2);

   private:
    MapManage(const std::string& map_position);

    int saveMapById(uint32_t id, const Map& map);

    std::string map_position_;  // maps存储的路径
    std::mutex mutex_;
};
}  // namespace netaos
}  // namespace hozon