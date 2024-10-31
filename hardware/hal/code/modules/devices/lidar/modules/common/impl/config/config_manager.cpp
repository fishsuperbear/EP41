#include "lidar/modules/common/impl/config/config_manager.h"

std::shared_ptr<ConfigManager> ConfigManager::instance_ = nullptr;
std::mutex ConfigManager::mutex_;

std::shared_ptr<ConfigManager> ConfigManager::getInstance()
{
    if (instance_ == nullptr)
    {
        std::unique_lock<std::mutex> lk(mutex_);
        if (instance_ == nullptr)
        {
            instance_.reset(new ConfigManager);
        }
    }
    return instance_;
}

ConfigManager::ConfigManager()
{
}

ConfigManager::~ConfigManager()
{
}

const std::vector<LidarConfig> ConfigManager::getLidarConfig(const struct hw_lidar_callback_t *i_callback)
{
    std::vector<LidarConfig> configs;
    for (int i = 0; i < i_callback->config_number; i++)
    {
        LidarConfig config;
        config.index = i_callback->configs[i].index;
        config.lidar_model = i_callback->configs[i].lidar_model;
        config.port = i_callback->configs[i].port;
        switch (config.lidar_model)
        {
        case HW_LIDAR_MODEL::ROBOSENSE_M1:
            config.packets_per_frame = 630;
            config.points_per_frame = 78750;
            break;
        default:
            HW_LIDAR_LOG_ERR("get config failed! unknown lidar model: %d\n", config.lidar_model);
            continue;
        }

        configs.push_back(config);
    }
    return configs;
}
