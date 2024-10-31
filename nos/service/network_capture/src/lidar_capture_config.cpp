#include "network_capture/include/lidar_capture_config.h"
#include "network_capture/include/lidar_struct_define.h"

namespace hozon {
namespace netaos {
namespace network_capture {

std::unique_ptr<LidarFilterInfo> LidarFilterInfo::LoadConfig() {
    auto cfg_ptr = std::make_unique<LidarFilterInfo>();
#ifdef BUILD_FOR_ORIN
    cfg_ptr->eth_name = "mgbe3_0.80";
#else
    cfg_ptr->eth_name = "enp0s31f6";
#endif
    cfg_ptr->src_port = "58005";
    cfg_ptr->dst_port = "2368";
    cfg_ptr->src_host = "172.16.80.20";
    cfg_ptr->dst_host = "239.255.0.1";
    return cfg_ptr;
}
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon