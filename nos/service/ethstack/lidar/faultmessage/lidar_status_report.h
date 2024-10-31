#pragma once 

#include <cstdint>
#include "common/logger.h"
#include "third_party/orin/rocketmq/include/RocketMQClient.h"


namespace hozon {
namespace ethstack {
namespace lidar {



class LidarStatusReport {
public:
    static LidarStatusReport& Instance();
    ~LidarStatusReport(){};

    uint8_t GetLidarStatus();
    bool SetLidarStatus(uint8_t lidar_status);

private:
    LidarStatusReport();
    uint8_t lidar_status_;
    
};




}
}
}