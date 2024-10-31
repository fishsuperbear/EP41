#include "lidar_status_report.h"
#include <cstdint>
#include <string>



namespace hozon {
namespace ethstack {
namespace lidar {

LidarStatusReport::LidarStatusReport()    
: lidar_status_(0)
{
}

LidarStatusReport& LidarStatusReport::Instance() {
    static LidarStatusReport instance;
    return instance;
}

uint8_t LidarStatusReport::GetLidarStatus() {
    return lidar_status_;
}

bool LidarStatusReport::SetLidarStatus(uint8_t lidar_status) {
    lidar_status_ = lidar_status;
    // if (lidar_status_ != lidar_status) {
    //     return false;
    // }
    return true;
}


}
}
}