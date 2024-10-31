#ifndef HW_LIDAR_SOCKET_CONFIG_MANAGER_H
#define HW_LIDAR_SOCKET_CONFIG_MANAGER_H

#include <iostream>
#include <vector>
#include <mutex>
#include <memory>

#include "lidar/modules/common/hw_lidar_log_impl.h"
#include "lidar/modules/common/impl/utils/lidar_types.h"

class ConfigManager
{
public:
    ConfigManager();
    ~ConfigManager();

    const std::vector<LidarConfig> getLidarConfig(const struct hw_lidar_callback_t *i_callback);

public:
    static std::shared_ptr<ConfigManager> getInstance();

    static std::mutex mutex_;
    static std::shared_ptr<ConfigManager> instance_;
};

#endif // HW_LIDAR_SOCKET_CONFIG_MANAGER_H
