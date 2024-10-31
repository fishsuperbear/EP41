#include "hal_lidar.hpp"

#include "lidar.h"

namespace hal {
namespace lidar {

Lidar lidar_;

ErrorCode Start(const std::vector<ConfigInfo> &configs, PointsCallback callback) {
    printf("start hal_lidar\n");
    ErrorCode res = lidar_.Start(configs, callback);
    if (res != ErrorCode::SUCCESS) {
        printf("start lidar failed!\n");
        return res;
    }

    return ErrorCode::SUCCESS;
}

void Stop() {
    printf("stop hal_lidar\n");
    lidar_.Stop();
}

}  // namespace lidar
}  // namespace hal