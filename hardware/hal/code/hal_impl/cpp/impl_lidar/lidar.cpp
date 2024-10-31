#include "lidar.h"

Lidar::Lidar() {
}

Lidar::~Lidar() {
    Stop();
}

ErrorCode Lidar::Start(const std::vector<ConfigInfo> &configs, PointsCallback callback) {
    running_flag_ = true;
    config_info_vec_ = configs;
    points_callback_ = callback;

    for (int i = 0; i < config_info_vec_.size(); i++) {
        ConfigInfo config = config_info_vec_[i];
        if (config.model != "hesai_at128") {
            printf("lidar%d model: %s is not support!\n", config.index, config.model.c_str());
            Stop();
            return ErrorCode::UNKNOWN_LIDAR_MODEL;
        }

        p_thread_map_[config.index] = std::make_shared<std::thread>(&Lidar::HesaiSdkThreadHandle, this, config);
    }

    return ErrorCode::SUCCESS;
}

void Lidar::Stop() {
    running_flag_ = false;
    for (ConfigInfo config : config_info_vec_) {
        int index = config.index;
        pandar_swift_sdk_map_[index]->stop();
        pandar_swift_sdk_map_[index] = nullptr;

        if (p_thread_map_[index]->joinable()) {
            p_thread_map_[index]->join();
        }
        p_thread_map_[index] = nullptr;
    }
}

void Lidar::HesaiSdkThreadHandle(const ConfigInfo &config) {
    std::map<std::string, int32_t> hesaiConfigMap;
    hesaiConfigMap["process_thread"] = 91;
    hesaiConfigMap["publish_thread"] = 90;
    hesaiConfigMap["read_thread"] = 99;
    hesaiConfigMap["timestamp_num"] = 0;
    hesaiConfigMap["without_pointcloud_udp_warning_time"] = 10000;    // ms
    hesaiConfigMap["without_faultmessage_udp_warning_time"] = 10000;  // ms
    hesaiConfigMap["untragger_pclcallback_warning_time"] = 10000;     // ms

    pandar_swift_sdk_map_[config.index].reset(new PandarSwiftSDK(
        std::string(config.ip), std::string(""), config.port, 10110, std::string("PandarAT128"),
        std::string(""),
        std::string(""),
        std::string(""),
        boost::bind(&Lidar::OnPointcloudCallback, this, 
                    boost::placeholders::_1, 
                    boost::placeholders::_2, 
                    boost::placeholders::_3, 
                    boost::placeholders::_4),
        NULL,
        NULL,
        NULL,
        std::string(""),
        std::string(""),
        std::string(""),
        0, 0, 1,
        std::string("both_point_raw"),
        std::string(""),  // multicast ip
        config, 
        hesaiConfigMap));
    pandar_swift_sdk_map_[config.index]->start();

    while (running_flag_) {
        sleep(1);
    }
}

void Lidar::OnPointcloudCallback(boost::shared_ptr<PPointCloud> cld, double timestamp, uint32_t seq, hal::lidar::ConfigInfo config) {
    std::unique_lock<std::mutex> lk(mutex_);

    PointsCallbackData points_callback_data;
    points_callback_data.index = config.index;
    points_callback_data.model = config.model;
    points_callback_data.sequence = seq;
    points_callback_data.timestamp = timestamp * 1e9;
    points_callback_data.points.swap(*cld.get());
    if (points_callback_) {
        points_callback_(points_callback_data);
    }
}
