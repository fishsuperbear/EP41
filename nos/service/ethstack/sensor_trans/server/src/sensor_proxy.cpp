#include <memory>
#include <thread>
#include <cstddef>
#include "logger.h"
#include "sensor_proxy.h"
#include "proxy_imu_ins.h"
#include "ara/com/sample_ptr.h"

namespace hozon {
namespace netaos {
namespace sensor {

SensorProxy::SensorProxy() {
    _imuins_proxy = std::make_shared<ImuInsProxy>();
    _chassis_proxy = std::make_shared<ChassisProxy>();
    _gnss_proxy = std::make_shared<ProxyGnss>();
    _mcu2ego_proxy = std::make_shared<Mcu2EgoProxy>();
    _pnc_ctr_proxy = std::make_shared<PncCtrProxy>();
    _uss_proxy = std::make_shared<UssProxy>();

    _radarfront_proxy = std::make_shared<RadarProxy>(std::string("radarfront"));
    _radarcorner1_proxy = std::make_shared<RadarProxy>(std::string("radarcorner1"));
    _radarcorner2_proxy = std::make_shared<RadarProxy>(std::string("radarcorner2"));
    _radarcorner3_proxy = std::make_shared<RadarProxy>(std::string("radarcorner3"));
    _radarcorner4_proxy = std::make_shared<RadarProxy>(std::string("radarcorner4"));
}
SensorProxy::~SensorProxy() {

}
int SensorProxy::Init() {
    SENSOR_LOG_INFO << "sensor Proxy Init.";
    _chassis_proxy->Init();
    _radarcorner1_proxy->Init();
    _radarcorner2_proxy->Init();
    _radarcorner3_proxy->Init();
    _radarcorner4_proxy->Init();
    _radarfront_proxy->Init();
    
    ara::core::Initialize();

    
    
    
    // std::future<void> future = promise->get_future();
    SENSOR_LOG_INFO << "McuDataServiceProxy Startfindservice.";
    find_McuDataService_handle_ = McuDataServiceProxy::StartFindService(
        [this](ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handle) {
            SENSOR_LOG_INFO << "McuDataServiceProxy StartFindService service available callback," << "handles.size() = " << handles.size();
            if (handles.size() > 0U) {
                std::lock_guard<std::mutex> lock(proxy_mutex_);
                if (!McuDataServiceProxy_) {
                    McuDataServiceProxy_ = std::make_shared<McuDataServiceProxy>(handles[0]);
                    HanleSomeIpData();
                }
            } else {
                McuDataServiceProxy_ = nullptr;
                SENSOR_LOG_INFO << "McuDataServiceProxy is nullptr";
            }
        },
        ara::com::InstanceIdentifier("1"));
    SENSOR_LOG_INFO << "McuFrontRadarServiceProxy Startfindservice.";
     find_McuFrontRadarService_handle_ = McuFrontRadarServiceProxy::StartFindService(
        [this](ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handle) {
            SENSOR_LOG_INFO << "McuFrontRadarServiceProxy StartFindService service available callback.";
            if (handles.size() > 0U) {
                std::lock_guard<std::mutex> lock(proxy_mutex_);
                if (!McuFrontRadarServiceProxy_) {
                    McuFrontRadarServiceProxy_ = std::make_shared<McuFrontRadarServiceProxy>(handles[0]);
                    HanleSomeIpData();
                }
            } else {
                McuFrontRadarServiceProxy_ = nullptr;
                SENSOR_LOG_INFO << "McuFrontRadarServiceProxy is nullptr.";
            }
        },
        ara::com::InstanceIdentifier("1"));

    SENSOR_LOG_INFO << "McuCornerRadarServiceProxy Startfindservice.";
     find_McuCornerRadarService_handle_ = McuCornerRadarServiceProxy::StartFindService(
        [this](ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handle) {
            SENSOR_LOG_INFO << "McuCornerRadarServiceProxy StartFindService service available callback." ;
            if (handles.size() > 0U) {
                std::lock_guard<std::mutex> lock(proxy_mutex_);
                if (!McuCornerRadarServiceProxy_) {
                    McuCornerRadarServiceProxy_ = std::make_shared<McuCornerRadarServiceProxy>(handles[0]);
                    HanleSomeIpData();
                }
            } else {
                McuCornerRadarServiceProxy_ = nullptr;
                SENSOR_LOG_INFO << "McuCornerRadarServiceProxy is nullptr.";
            }
        },
        ara::com::InstanceIdentifier("1"));

    return 0;
}

void SensorProxy::Deinit() {
    SENSOR_LOG_INFO << "sensor Proxy Deinit.";
    if(McuDataServiceProxy_) {
        McuDataServiceProxy_->AlgImuInsInfo.Unsubscribe();
        McuDataServiceProxy_->AlgChassisInfo.Unsubscribe();
        McuDataServiceProxy_->AlgGNSSPosInfo.Unsubscribe();
        McuDataServiceProxy_->AlgMcuToEgo.Unsubscribe();
        McuDataServiceProxy_->AlgPNCControl.Unsubscribe();
        McuDataServiceProxy_->AlgUssRawdata.Unsubscribe();
    }
    
    if(McuFrontRadarServiceProxy_) {
        McuFrontRadarServiceProxy_->AlgFrontRadarTrack.Unsubscribe();
    }
    
    if(McuCornerRadarServiceProxy_) {
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFR.Unsubscribe();
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFL.Unsubscribe();
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRR.Unsubscribe();
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRL.Unsubscribe();
    }
    
    McuDataServiceProxy::StopFindService(find_McuDataService_handle_);
    McuFrontRadarServiceProxy::StopFindService(find_McuFrontRadarService_handle_);
    McuCornerRadarServiceProxy::StopFindService(find_McuCornerRadarService_handle_);

    ara::core::Deinitialize();
    _chassis_proxy->DeInit();
    SENSOR_LOG_INFO << "sensor Proxy Deinit successful.";
}

void SensorProxy::HanleSomeIpData() {
    SENSOR_LOG_INFO << "start HanleSomeIpData.";
    if (McuDataServiceProxy_) {
        SENSOR_LOG_INFO << "start HanleSomeIpData:McuDataServiceProxy_.";
        // receive someip imuins
        McuDataServiceProxy_->AlgImuInsInfo.SetReceiveHandler([this]() {
            if (McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgImuInsInfo.GetNewSamples(
                    [this](ara::com::SamplePtr<::hozon::netaos::AlgImuInsInfo const> sample) {
                        Write("imuins", _imuins_proxy->Trans(sample));
                    },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set Imuins Receive Handler";
        });
        McuDataServiceProxy_->AlgImuInsInfo.Subscribe(10);

        // receive someip chassis
        McuDataServiceProxy_->AlgChassisInfo.SetReceiveHandler([this]() {
            if(McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgChassisInfo.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const>
                            sample) {
                            Write("chassis", _chassis_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set Chassis Receive Handler";
        });
        McuDataServiceProxy_->AlgChassisInfo.Subscribe(10);

        // receive someip gnss
        McuDataServiceProxy_->AlgGNSSPosInfo.SetReceiveHandler([this]() {
            if(McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgGNSSPosInfo.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgGnssInfo const>
                            sample) { 
                                Write("gnss", _gnss_proxy->Trans(sample));   
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set GNSS Receive Handler";
        });
        McuDataServiceProxy_->AlgGNSSPosInfo.Subscribe(10);

        // receive someip mcutoego
        McuDataServiceProxy_->AlgMcuToEgo.SetReceiveHandler([this]() {
            if(McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgMcuToEgo.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgMcuToEgoFrame const>
                            sample) {
                            Write("mcu2ego", _mcu2ego_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set McuToEgo Receive Handler";
        });
        McuDataServiceProxy_->AlgMcuToEgo.Subscribe(10);

        // receive someippcn
        McuDataServiceProxy_->AlgPNCControl.SetReceiveHandler([this]() {
            if(McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgPNCControl.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::PNCControlState const>
                            sample) {
                            Write("pnc", _pnc_ctr_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set PNCControl Receive Handler";
        });
        McuDataServiceProxy_->AlgPNCControl.Subscribe(10);

        // receive someip uss
        McuDataServiceProxy_->AlgUssRawdata.SetReceiveHandler([this]() {
            if(McuDataServiceProxy_) {
                McuDataServiceProxy_->AlgUssRawdata.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::UssRawDataSet const>
                            sample) {
                            Write("uss", _uss_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuDataServiceProxy set ussrawdata Receive Handler";    
        });
        McuDataServiceProxy_->AlgUssRawdata.Subscribe(10);
    }

    if (McuFrontRadarServiceProxy_) {
        SENSOR_LOG_INFO << "start HanleSomeIpData:McuFrontRadarServiceProxy_.";
        // receive  front radar
        McuFrontRadarServiceProxy_->AlgFrontRadarTrack.SetReceiveHandler([this]() {
            if (McuFrontRadarServiceProxy_) {
                McuFrontRadarServiceProxy_->AlgFrontRadarTrack.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgFrontRadarTrackArrayFrame const>
                            sample) {
                            Write("radarfront", _radarfront_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuRrontRadarServiceProxy set radarfront Receive Handler";
        });
        McuFrontRadarServiceProxy_->AlgFrontRadarTrack.Subscribe(10);
    }

    if (McuCornerRadarServiceProxy_) {
        SENSOR_LOG_INFO << "start HanleSomeIpData:McuCornerRadarServiceProxy_.";
        // receive  corner radar
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFR.SetReceiveHandler([this]() {
            if(McuCornerRadarServiceProxy_) {
                McuCornerRadarServiceProxy_->AlgCornerRadarTrackFR.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const>
                            sample) {
                            Write("radarcorner1", _radarcorner1_proxy->Trans(sample));            
                        },
                    10);
            }
            // SENSOR_LOG_INFO << "McuCornerRadarServiceProxy set radarcorner1 Receive Handler";
        });
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFR.Subscribe(10);

        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFL.SetReceiveHandler([this]() {
            if(McuCornerRadarServiceProxy_) {
                McuCornerRadarServiceProxy_->AlgCornerRadarTrackFL.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const>
                            sample) {
                            Write("radarcorner2", _radarcorner2_proxy->Trans(sample));            
                        },
                    10);
            }
        });
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackFL.Subscribe(10);

        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRR.SetReceiveHandler([this]() {
            if(McuCornerRadarServiceProxy_) {
                McuCornerRadarServiceProxy_->AlgCornerRadarTrackRR.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const>
                            sample) {
                            Write("radarcorner3", _radarcorner3_proxy->Trans(sample));            
                        },
                    10);
            }
        });
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRR.Subscribe(10);

        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRL.SetReceiveHandler([this]() {
            if(McuCornerRadarServiceProxy_) {
                McuCornerRadarServiceProxy_->AlgCornerRadarTrackRL.GetNewSamples(
                    [this](
                        ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const>
                            sample) {
                            Write("radarcorner4", _radarcorner4_proxy->Trans(sample));            
                        },
                    10);
            }
        });
        McuCornerRadarServiceProxy_->AlgCornerRadarTrackRL.Subscribe(10);
    }

}

int SensorProxy::Run() {
    // ara::com::SamplePtr<const hozon::netaos::UssRawDataSet> *uss 
    //     = new ara::com::SamplePtr<const hozon::netaos::UssRawDataSet>();
    // Write("uss", _uss_proxy->Trans((std::make_shared<hozon::netaos::UssRawDataSet>())));
    // ara::com::SamplePtr<::hozon::netaos::AlgImuInsInfo const> imuins;
    // Write("imuins", _imuins_proxy->Trans(imuins));

    // ara::com::SamplePtr<::hozon::netaos::AlgChassisInfo const> chassis;
    // Write("chassis", _chassis_proxy->Trans(chassis));

    // ara::com::SamplePtr<::hozon::netaos::AlgGnssInfo const> gnss;
    // Write("gnss", _gnss_proxy->Trans(gnss)); 

    // ara::com::SamplePtr<::hozon::netaos::AlgMcuToEgoFrame const> mcu2ego;
    // Write("mcu2ego", _mcu2ego_proxy->Trans(mcu2ego));    

    // ara::com::SamplePtr<::hozon::netaos::PNCControlState const> pnc;
    // Write("pnc", _pnc_ctr_proxy->Trans(pnc));

    // // 毫米波雷达
    // ara::com::SamplePtr<::hozon::netaos::AlgFrontRadarTrackArrayFrame const> radarfront;
    // Write("radarfront", _radarfront_proxy->Trans(radarfront));

    // ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const> radarcorner1;
    // Write("radarcorner1", _radarcorner1_proxy->Trans(radarcorner1));

    // ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const> radarcorner2;
    // Write("radarcorner2", _radarcorner2_proxy->Trans(radarcorner2));

    // ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const> radarcorner3;
    // Write("radarcorner3", _radarcorner3_proxy->Trans(radarcorner3));

    // ara::com::SamplePtr<::hozon::netaos::AlgCornerRadarTrackArrayFrame const> radarcorner4;
    // Write("radarcorner4", _radarcorner4_proxy->Trans(radarcorner4));

    return 0;
}

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon