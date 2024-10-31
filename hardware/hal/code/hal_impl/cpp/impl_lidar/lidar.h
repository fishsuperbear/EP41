#ifndef _HAL_IMPL_CPP_LIDAR_H_
#define _HAL_IMPL_CPP_LIDAR_H_

#include <sys/wait.h>

#include <map>
#include <memory>
#include <thread>
#include <mutex>

#include "hal_lidar.hpp"
#include "sdk/hesai/include/pandarSwiftSDK.h"

using namespace hal::lidar;

class Lidar {
public:
    Lidar();
    ~Lidar();

    ErrorCode Start(const std::vector<ConfigInfo> &configs, PointsCallback callback);
    void Stop();

private:
    void HesaiSdkThreadHandle(const ConfigInfo &config);
    void OnPointcloudCallback(boost::shared_ptr<PPointCloud> cld, double timestamp, uint32_t seq, hal::lidar::ConfigInfo config);

    std::mutex mutex_;
    bool running_flag_ = true;
    std::vector<ConfigInfo> config_info_vec_;
    PointsCallback points_callback_;
    std::map<int, std::shared_ptr<std::thread>> p_thread_map_;
    std::map<int, boost::shared_ptr<PandarSwiftSDK>> pandar_swift_sdk_map_;
};

#endif  // _HAL_IMPL_CPP_LIDAR_H_
