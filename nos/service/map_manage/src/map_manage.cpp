#include "map_manage.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <regex>
#include <string>
#include "orin_log.h"
#include "yaml-cpp/yaml.h"

// #include "common_log.h"
// #define HZ_MM_LOG_ERROR PRINTLN << "[MAP_MANAGE][ERROR] "
// #define HZ_MM_LOG_WARN PRINTLN << "[MAP_MANAGE][WARN] "
// #define HZ_MM_LOG_INFO PRINTLN << "[MAP_MANAGE][INFO] "
// #define HZ_MM_LOG_DEBUG PRINTLN << "[MAP_MANAGE][DEBUG] "
// #define HZ_MM_LOG_TRACE PRINTLN << "[MAP_MANAGE][TRACE] "

namespace hozon {
namespace netaos {

MapManage::MapManage(const std::string& map_position) : map_position_(map_position) {
    hozon::netaos::HZMMLogger::GetInstance().InitLogging();
}

int MapManage::saveMapById(uint32_t id, const Map& map) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();  // 计算耗时

    std::string map_dir_path = map_position_ + "map_" + std::to_string(id) + "/";
    // desc
    std::string yaml_path = map_dir_path + "desc.yaml";
    // planning
    std::string map_db_path = map_dir_path + "map.db";
    std::string path_db_path = map_dir_path + "path.db";
    // slam
    std::string slam_map_db_path = map_dir_path + "slam_map.db";
    std::string slam_path_db_path = map_dir_path + "slam_path.db";
    std::string feature_map_db_path = map_dir_path + "feature_map.db";

    // 首先判断本地是否已经存在相同id
    if (access(map_dir_path.c_str(), F_OK) == 0) {
        HZ_MM_LOG_ERROR << "the folder corresponding to this ID already exists";
        return -1;
    } else {
        int ret = mkdir(map_dir_path.c_str(), 0777);
        if (ret != 0) {
            HZ_MM_LOG_ERROR << ret << " create the map folder fail";
            return -1;
        }
    }

    // save desc.yaml
    {
        auto header = map.map_planning.map.header();
        double longitude = header.j02longitude();
        double latitude = header.j02latitude();

        auto save_time = std::chrono::system_clock::now();                          // 获取系统时间
        std::time_t save_time_t = std::chrono::system_clock::to_time_t(save_time);  // 转换为以秒为单位的时间戳
        auto save_time_str = std::ctime(&save_time_t);                              // 将 time_t 类型转换为字符串
        HZ_MM_LOG_DEBUG << "the current save time is " << save_time_str;

        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "id" << YAML::Value << id;
        out << YAML::Key << "longitude" << YAML::Value << longitude;
        out << YAML::Key << "latitude" << YAML::Value << latitude;
        out << YAML::Key << "save_time" << YAML::Value << save_time_str;
        out << YAML::EndMap;

        std::fstream yaml_out(yaml_path.c_str(), std::ios::out | std::ios::binary);
        if (!yaml_out) {
            HZ_MM_LOG_ERROR << "save desc.yaml file fail : " << yaml_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        yaml_out << out.c_str();
    }

    // save map.db
    {
        std::fstream map_output(map_db_path.c_str(), std::ios::out | std::ios::binary);
        if (!map_output) {
            HZ_MM_LOG_ERROR << "save map.db fail : " << map_db_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        map.map_planning.map.SerializeToOstream(&map_output);
    }

    // save path.db
    {
        std::fstream path_output(path_db_path.c_str(), std::ios::out | std::ios::binary);
        if (!path_output) {
            HZ_MM_LOG_ERROR << "save path.db fail :  " << path_db_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        map.map_planning.path.SerializeToOstream(&path_output);
    }

    // save slam_map.db
    {
        std::fstream map_output(slam_map_db_path.c_str(), std::ios::out | std::ios::binary);
        if (!map_output) {
            HZ_MM_LOG_ERROR << "save slam_map.db fail : " << slam_map_db_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        map.map_slam.map.SerializeToOstream(&map_output);
    }

    // save slam_path.db
    {
        std::fstream path_output(slam_path_db_path.c_str(), std::ios::out | std::ios::binary);
        if (!path_output) {
            HZ_MM_LOG_ERROR << "save slam_path.db fail :  " << slam_path_db_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        map.map_slam.path.SerializeToOstream(&path_output);
    }

    // save feature_map.db
    {
        std::fstream map_output(feature_map_db_path.c_str(), std::ios::out | std::ios::binary);
        if (!map_output) {
            HZ_MM_LOG_ERROR << "save feature_map.db fail : " << feature_map_db_path << " is not found.";
            deleteMap(id);
            return -1;
        }
        map_output << map.feature_map;
    }

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "saveMap() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return 0;
}

int MapManage::saveMap(const Map& map) {
    // 获取本地已有的map id
    std::vector<uint32_t> maps = pollAllMap();  //是否需要去重
    if (maps.size() >= 99) {
        HZ_MM_LOG_ERROR << "too many maps, max 100";
        return -1;
    }

    // 先维护一个长度为100的标记数组，有数据为true，无数据为false，然后从左边找到第一个false的位置
    uint32_t id = 0;
    std::vector<bool> ids(100, false);

    for (auto& map : maps) {
        if (map > 0 && map < 100) {
            ids[map] = true;
        }
    }
    // 座舱那边0为默认值，id从1开始计数
    for (uint32_t i = 1; i < 100; i++) {
        if (!ids[i]) {
            id = i;
            int ret = saveMapById(id, map);
            if (ret < 0) {
                return -1;
            } else {
                break;
            }
        }
    }

    return id;
}

int MapManage::updateMap(uint32_t id, const Map& map) {
    if (deleteMap(id) == -1) {
        HZ_MM_LOG_ERROR << "update map fail, file not found";
        return -1;
    }

    int ret = saveMapById(id, map);
    if (ret < 0) {
        HZ_MM_LOG_ERROR << "update map fail.";
        return -1;
    }

    return 0;
}

int MapManage::setMapId(uint32_t select_id) {
    std::lock_guard<std::mutex> locker(mutex_);

    // map.yaml
    std::string map_yaml_path = map_position_ + "map.yaml";
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "select_id" << YAML::Value << select_id;
    out << YAML::EndMap;

    std::fstream yaml_out(map_yaml_path.c_str(), std::ios::out | std::ios::binary);
    if (!yaml_out) {
        HZ_MM_LOG_ERROR << "save map.yaml file fail : " << map_yaml_path << " is not found.";
        return -1;
    }
    yaml_out << out.c_str();
    return 0;
}

uint32_t MapManage::getMapId() {
    std::lock_guard<std::mutex> locker(mutex_);

    // map.yaml
    std::string map_yaml_path = map_position_ + "map.yaml";
    std::ifstream fin(map_yaml_path);
    if (!fin) {
        HZ_MM_LOG_ERROR << "fail to open the map.yaml file.";
        return 0;  // 这里的0表示返回默认无效id
    }
    YAML::Node config = YAML::Load(fin);
    uint32_t select_id = config["select_id"].as<uint32_t>(0);

    return select_id;
}

std::shared_ptr<MapManage::MapContent> MapManage::getMapPlanning(uint32_t id) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::string map_dir_path = map_position_ + "map_" + std::to_string(id) + "/";
    std::string map_db_path = map_dir_path + "map.db";
    std::string path_db_path = map_dir_path + "path.db";

    auto context = std::make_shared<MapContent>();

    context->id = id;

    std::fstream input_1(map_db_path, std::ios::in | std::ios::binary);
    if (!input_1) {
        HZ_MM_LOG_ERROR << "map.db file " << map_db_path << " is not found.";
        return nullptr;
    }

    if (!context->map.ParseFromIstream(&input_1)) {
        HZ_MM_LOG_ERROR << "faild to parse map_obj.";
        return nullptr;
    }
    input_1.close();

    std::fstream input_2(path_db_path, std::ios::in | std::ios::binary);
    if (!input_2) {
        HZ_MM_LOG_ERROR << "path.db file " << path_db_path << " is not found.";
        return nullptr;
    }

    if (!context->path.ParseFromIstream(&input_2)) {
        HZ_MM_LOG_ERROR << "faild to parse path_obj.";
        return nullptr;
    }
    input_2.close();

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "getMapPlanning() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return context;
}

std::shared_ptr<MapManage::MapContent> MapManage::getMapPlanning() {
    uint32_t select_id = getMapId();
    if (select_id <= 0) {
        HZ_MM_LOG_INFO << "get invalid select_id : " << select_id;
        return nullptr;
    }

    auto content = getMapPlanning(select_id);
    if (!content) {
        HZ_MM_LOG_INFO << "get map planning failed";
        return nullptr;
    }
    return content;
}

std::shared_ptr<MapManage::MapContent> MapManage::getMapSlam(uint32_t id) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::string map_dir_path = map_position_ + "map_" + std::to_string(id) + "/";
    std::string slam_map_db_path = map_dir_path + "slam_map.db";
    std::string slam_path_db_path = map_dir_path + "slam_path.db";

    auto context = std::make_shared<MapContent>();

    context->id = id;

    // read slam_map.db
    std::fstream input_1(slam_map_db_path, std::ios::in | std::ios::binary);
    if (!input_1) {
        HZ_MM_LOG_ERROR << "slam_map.db file " << slam_map_db_path << " is not found.";
        return nullptr;
    }

    if (!context->map.ParseFromIstream(&input_1)) {
        HZ_MM_LOG_ERROR << "faild to parse slam_map_obj.";
        return nullptr;
    }
    input_1.close();

    // read slam_path.db
    std::fstream input_2(slam_path_db_path, std::ios::in | std::ios::binary);
    if (!input_2) {
        HZ_MM_LOG_ERROR << "slam_path.db file " << slam_path_db_path << " is not found.";
        return nullptr;
    }

    if (!context->path.ParseFromIstream(&input_2)) {
        HZ_MM_LOG_ERROR << "faild to parse path_obj.";
        return nullptr;
    }
    input_2.close();

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "getMapSlam() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return context;
}

std::shared_ptr<MapManage::MapContent> MapManage::getMapSlam() {
    uint32_t select_id = getMapId();
    if (select_id <= 0) {
        HZ_MM_LOG_INFO << "get invalid select_id : " << select_id;
        return nullptr;
    }

    auto content = getMapSlam(select_id);
    if (!content) {
        HZ_MM_LOG_INFO << "get map slam failed";
        return nullptr;
    }
    return content;
}

std::shared_ptr<std::string> MapManage::getFeatureMap(uint32_t id) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::string map_dir_path = map_position_ + "map_" + std::to_string(id) + "/";
    std::string feature_map_db_path = map_dir_path + "feature_map.db";

    // read feature_map.db
    std::fstream input(feature_map_db_path, std::ios::in | std::ios::binary);
    if (!input) {
        HZ_MM_LOG_ERROR << "feature_map.db file " << feature_map_db_path << " is not found.";
        return nullptr;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();

    auto context = std::make_shared<std::string>(buffer.str());
    if (!context->size()) {
        HZ_MM_LOG_ERROR << "feature_map.db file is empty.";
        return nullptr;
    }

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "getFeatureMap() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return context;
}

std::shared_ptr<std::string> MapManage::getFeatureMap() {
    uint32_t select_id = getMapId();
    if (select_id <= 0) {
        HZ_MM_LOG_INFO << "get invalid select_id : " << select_id;
        return nullptr;
    }

    auto content = getFeatureMap(select_id);
    if (!content) {
        HZ_MM_LOG_INFO << "get feature map failed";
        return nullptr;
    }
    return content;
}

int MapManage::deleteMap(uint32_t id) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::string map_dir_path = map_position_ + "map_" + std::to_string(id) + "/";
    std::string command = "rm -r ";
    command += map_dir_path;
    int ret = std::system(command.c_str());

    if (ret != 0) {
        HZ_MM_LOG_ERROR << ret << " fail to delete the directory.";
        return -1;
    }

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "deleteMap() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return 0;
}

std::pair<double, double> MapManage::getXY(uint32_t id) {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::string yaml_path = map_position_ + "map_" + std::to_string(id) + "/desc.yaml";
    HZ_MM_LOG_DEBUG << "yaml_path" << yaml_path;

    // 从yaml文件读取经纬度
    std::ifstream fin(yaml_path);
    if (!fin) {
        HZ_MM_LOG_ERROR << "fail to open the yaml file.";
        return {0, 0};
    }

    YAML::Node config = YAML::Load(fin);

    uint32_t map_id = config["id"].as<uint32_t>(0);
    if (map_id != id) {
        HZ_MM_LOG_ERROR << "id not match in the yaml file.";
        return {0, 0};
    }

    double longitude = config["longitude"].as<double>();
    double latitude = config["latitude"].as<double>();
    HZ_MM_LOG_DEBUG << "id: " << id << " longitude: " << longitude << " latitude: " << latitude;

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "getXY() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return {longitude, latitude};
}

std::vector<uint32_t> MapManage::pollAllMap() {
    std::lock_guard<std::mutex> locker(mutex_);
    auto now = std::chrono::steady_clock::now();

    std::vector<uint32_t> all_map_id{};

    DIR* dir = opendir(map_position_.c_str());
    if (dir == nullptr) {
        HZ_MM_LOG_ERROR << "failed to open the folder: " << map_position_;
        return {};
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // 排除当前目录和父目录
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // 提取子文件夹的id信息
        std::string folder_name = entry->d_name;
        HZ_MM_LOG_DEBUG << "folder_name: " << folder_name;

        std::regex pattern("map_(\\d+)");
        std::smatch match;
        if (std::regex_search(folder_name, match, pattern)) {
            std::string id_str = match[1].str();
            int id = std::stoi(id_str);
            HZ_MM_LOG_DEBUG << "folder_name_id: " << id;
            if (id > 0 && id < 100) {
                all_map_id.push_back(id);
            }
        }
    }
    closedir(dir);

    auto end = std::chrono::steady_clock::now();
    HZ_MM_LOG_INFO << "pollAllMap() used time:" << std::chrono::duration<double, std::milli>(end - now).count();

    return all_map_id;
}

double MapManage::calculateDistance(double lat1, double lon1, double lat2, double lon2) {
    // 地球平均半径（单位：米）
    const double EarthRadius = 6371000;

    // 将经纬度转换为弧度
    double lat1Rad = toRadians(lat1);
    double lon1Rad = toRadians(lon1);
    double lat2Rad = toRadians(lat2);
    double lon2Rad = toRadians(lon2);

    // 使用余弦定理计算球面角
    double angle = acos(sin(lat1Rad) * sin(lat2Rad) + cos(lat1Rad) * cos(lat2Rad) * cos(lon2Rad - lon1Rad));

    // 计算大圆距离
    double distance = EarthRadius * angle;

    return distance;
}
}  // namespace netaos
}  // namespace hozon