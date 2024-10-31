#pragma once


#include <algorithm>
#include <memory>
#include <mutex>
#include "hozon/netaos/v1/socdataservice_skeleton.h"
#include "skeleton_ego2mcu_chassis.h"
#include "skeleton_apa2mcu_chassis.h"
#include "skeleton_lane_detection.h"
#include "skeleton_location.h"
#include "skeleton_fusion.h"
#include "skeleton_apa_statemachine.h"
#include "cfg/include/config_param.h"


namespace hozon {
namespace netaos {
namespace sensor {
class SomeipSkeleton {
    using Skeleton = hozon::netaos::v1::skeleton::SocDataServiceSkeleton;
public:
    SomeipSkeleton();
    ~SomeipSkeleton() = default;
    int Init(uint32_t is_nnp);
    int Write(std::string name, adf::NodeBundle data);
    void Deinit();
    
private:
    void RunningMode(const std::string& client, const std::string& key, const uint8_t& value);
    int SomeipSendThread(std::string name, uint32_t interval);
    std::shared_ptr<Skeleton> _skeleton;
    std::shared_ptr<SkeletonEgo2McuChassis> _skeleton_ego2mcu_chassis;
    std::shared_ptr<SkeletonApa2McuChassis> _skeleton_apa2mcu_chassis;
    std::shared_ptr<SkeletonLaneDetection> _skeleton_lane_detection;
    std::shared_ptr<SkeletonLocation> _skeleton_location; // <std::string, adf::NodeBundle>
    std::shared_ptr<SkeletonFusion> _skeleton_fusion; // <std::string, adf::NodeBundle>
    std::shared_ptr<SkeletonApaStateMachine> _skeleton_apa_statemachime;
    hozon::netaos::AlgCanFdMsgFrame _apa2mcu_chassis_data;
    hozon::netaos::AlgEgoHmiFrame _ego2mcu_chassis_data;
    std::mutex _running_mode_mutex;
    std::recursive_mutex _apa2mcu_chassis_mutex;
    std::recursive_mutex _ego2mcu_chassis_mutex;
    uint8_t _running_mode;
    std::vector<std::shared_ptr<std::thread>> _send_thread; // <std::string, adf::NodeBundle>
    bool _need_stop;
    bool _init_ok;
};

}
}
}