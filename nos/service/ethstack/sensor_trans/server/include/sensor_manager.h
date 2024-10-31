#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <mutex>
#include "sensor_proxy_base.h"
#include "sensor_skeleton.h"
#include "chassis_server.h"
#include "someip_skeleton.h"
#include "cm_proxy.h"
#include "third_party/orin/fast-dds/include/fastdds/dds/core/policy/QosPolicies.hpp"

namespace hozon {
namespace netaos {
namespace sensor {
struct cm_info_map {
    std::string name;
    uint32_t domainID;
    std::string topic;
};

class SensorManager{
public:
    SensorManager() = default;
    ~SensorManager() = default;
    int Init(std::string &config, uint32_t is_nnp);
    int Run();
    int Stop();
    int Write(std::string name, std::shared_ptr<void> data);
    int32_t WriteSomeip(adf::NodeBundle* input, std::string name);
    void WaitStop(void);
    
private:
    std::shared_ptr<SensorProxyBase> CreateProxy();
    int InitRunningMode(uint32_t is_nnp);
    int ChangeRunningMode(bool is_init, uint8_t running_mode);
    std::unordered_map<std::string, std::shared_ptr<Skeleton>> _skeleton_instance_map;
    std::unordered_map<std::string, std::shared_ptr<SensorProxyBase>> _proxy_instance_map;
    std::unordered_map<std::string, std::shared_ptr<SomeipSkeleton>> _someip_skeleton_map;
    // std::unordered_map<std::string, std::shared_ptr<CmProxy>> _cm_proxy_map;
    std::shared_ptr<CmProxy> _cm_proxy;

    std::recursive_mutex _someip_skeleton_map_mt;
    std::shared_ptr<ChassisServer> _chassis_server;
};
}   // namespace sensor
}   // namespace netaos
}   // namespace hozon