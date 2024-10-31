#pragma  once

#include "proto/soc/chassis.pb.h"
#include "ara/com/sample_ptr.h"
#include "hozon/netaos/impl_type_algchassisinfo.h"
// #include "hozon/netaos/v1/mcudataservice_proxy.h"
#include "cfg/include/config_param.h"

namespace hozon {
namespace netaos {
namespace sensor {
    
#define kDEG2RAD (M_PI / 180)
class ChassisProxy {
public:
    ChassisProxy();
    ~ChassisProxy() = default;
    std::shared_ptr<hozon::soc::Chassis> Trans(
        ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const> data);
    int Init();
    int DeInit();

private:
    void PrintOriginalData(ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const> data);
    void TransToOta(ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const> data);
    hozon::soc::WheelSpeed::WheelSpeedType WheelSpeedTypeTrans(std::uint8_t data);
    void SetCfgParam(ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const> data);
    uint32_t chassis_seqid;
    uint64_t chassis_pub_last_time;
    uint32_t _chassis_pub_last_seq;
    // cfg::ConfigParam* _cfg_instance;
    uint8_t _last_power_mode;
};

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon
