#ifndef LIDAR_CAPTURE_CONFIG_H
#define LIDAR_CAPTURE_CONFIG_H
#pragma once

#include "network_capture/include/base_capture_config.h"


namespace hozon {
namespace netaos {
namespace network_capture {

class LidarFilterInfo : public BaseFilterInfo {
   public:
    
    // PandarATCorrections m_PandarAT_corrections;

    LidarFilterInfo()
    /* : m_PandarAT_corrections({}) */{ }

    ~LidarFilterInfo() = default;
    static std::unique_ptr<LidarFilterInfo> LoadConfig();
};


}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif