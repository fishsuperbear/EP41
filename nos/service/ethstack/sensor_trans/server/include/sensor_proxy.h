#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <ara/core/initialization.h>
#include "sensor_proxy_base.h"
#include "proxy_imu_ins.h"
#include "proxy_chassis.h"
#include "proxy_mcu2ego.h"
#include "proxy_gnss.h"
#include "proxy_pnc_control.h"
#include "proxy_radar.h"
#include "proxy_uss.h"

namespace hozon {
namespace netaos {
namespace sensor {

using McuDataServiceProxy = hozon::netaos::v1::proxy::McuDataServiceProxy;
using McuFrontRadarServiceProxy = hozon::netaos::v1::proxy::McuFrontRadarServiceProxy;
using McuCornerRadarServiceProxy = hozon::netaos::v1::proxy::McuCornerRadarServiceProxy;


class SensorProxy : public SensorProxyBase {
public:
    SensorProxy();
    ~SensorProxy();
    int Init();
    void Deinit();
private:
    int Run();
    void HanleSomeIpData();
    
    std::shared_ptr<ImuInsProxy> _imuins_proxy;
    std::shared_ptr<ChassisProxy> _chassis_proxy;
    std::shared_ptr<ProxyGnss> _gnss_proxy;
    std::shared_ptr<Mcu2EgoProxy> _mcu2ego_proxy;
    std::shared_ptr<PncCtrProxy> _pnc_ctr_proxy;
    std::shared_ptr<UssProxy> _uss_proxy;

    std::shared_ptr<RadarProxy> _radarfront_proxy;
    std::shared_ptr<RadarProxy> _radarcorner1_proxy;   // FR
    std::shared_ptr<RadarProxy> _radarcorner2_proxy;   // FL
    std::shared_ptr<RadarProxy> _radarcorner3_proxy;   // RR
    std::shared_ptr<RadarProxy> _radarcorner4_proxy;   // RL
 
    std::mutex proxy_mutex_;
    
    std::shared_ptr<McuDataServiceProxy> McuDataServiceProxy_;
    std::shared_ptr<McuFrontRadarServiceProxy> McuFrontRadarServiceProxy_;
    std::shared_ptr<McuCornerRadarServiceProxy> McuCornerRadarServiceProxy_;

    ara::com::FindServiceHandle find_McuDataService_handle_;
    ara::com::FindServiceHandle find_McuFrontRadarService_handle_;
    ara::com::FindServiceHandle find_McuCornerRadarService_handle_;

};
}   // namespace sensor
}   // namespace netaos
}   // namespace hozon

