#pragma once

#include <sys/param.h>
#include <cstdint>
#include <iostream>
#include <ostream>
#include "common.h"
#include "hozon/netaos/v1/mcucornerradarservice_proxy.h"
#include "hozon/netaos/v1/mcufrontradarservice_proxy.h"
#include "logger.h"
#include "proto/soc/radar.pb.h"
#include "param_config.h"

namespace hozon {
namespace netaos {
namespace sensor {

class RadarProxy {
   public:
   explicit RadarProxy(const std::string& name)
        : radar_pub_last_time(0u)
        , _radar_status(0)
        , _radar_lost_data_fualt_object(0u)
        , _radar_latency_fualt_object(0u)
        , _radar_allocat_get_fail_fualt_object(0u)
        , _radar_alive_monitor_checkpoint_id(0u)
        , _radar_pub_last_seq(0u) {
        _radar_name = name;
    };
    ~RadarProxy() = default;
    int Init() {
        SENSOR_LOG_INFO << _radar_name << " init.";
        if (_radar_name == "radarfront") {
            ParamConfig::GetInstance().GetParam("system/front_radar_status", _radar_status);
        }
        else if (_radar_name == "radarcorner1") {
            ParamConfig::GetInstance().GetParam("system/fr_radar_status", _radar_status);
        }
        else if (_radar_name == "radarcorner2") {
            ParamConfig::GetInstance().GetParam("system/fl_radar_status", _radar_status);
        }
        else if (_radar_name == "radarcorner3") {
            ParamConfig::GetInstance().GetParam("system/rr_radar_status", _radar_status);
        }
        else if (_radar_name == "radarcorner4") {
            ParamConfig::GetInstance().GetParam("system/rl_radar_status", _radar_status);
        }
        SENSOR_LOG_INFO << _radar_name << " system/radar_status " << _radar_status;
        return 0;
    }
    template <typename T>
    std::shared_ptr<hozon::soc::RadarTrackArrayFrame> Trans(ara::com::SamplePtr<T> data) {
        std::shared_ptr<hozon::soc::RadarTrackArrayFrame> radar_proto = std::make_shared<hozon::soc::RadarTrackArrayFrame>();

        if (nullptr == radar_proto) {
            SENSOR_LOG_ERROR << _radar_name << " Allocate got nullptr!";
            return nullptr;
        }

        radar_proto->mutable_header()->set_seq(data->header.seq);
        radar_proto->mutable_header()->set_frame_id(std::string(data->header.frameId.begin(), data->header.frameId.end()));

        if (!(radar_proto->mutable_header()->seq() % 100)) {
            struct timespec time;
            if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
                SENSOR_LOG_WARN << _radar_name << "clock_gettime fail ";
            }
            
            uint64_t current_time = (uint64_t)time.tv_sec * 1000lu + ((uint64_t)time.tv_nsec)/1000lu/1000lu;
            uint64_t time_diff = 0;
            if(radar_pub_last_time && (((time_diff = current_time - radar_pub_last_time) - 5000lu) > 500lu)) {
                SENSOR_LOG_WARN << "Send " << _radar_name << ", info seq: " << radar_proto->mutable_header()->seq() \
                    << " ,interval : " << time_diff << " ms";
            }
            else {
                SENSOR_LOG_INFO << "Send Send " << _radar_name << ", info seq: " << radar_proto->mutable_header()->seq() \
                    << " ,interval : " << time_diff << " ms";
            }
            radar_pub_last_time = current_time;
        }

        radar_proto->mutable_header()->set_publish_stamp(GetRealTimestamp());
        radar_proto->mutable_header()->set_gnss_stamp(HafTimeConverStamp(data->header.gnssStamp));
        radar_proto->mutable_header()->mutable_sensor_stamp()->set_radar_stamp(
            HafTimeConverStamp(data->header.stamp));
        radar_proto->set_sensor_id(data->sensorID);
        radar_proto->set_radar_state(data->radarState);
        radar_proto->set_is_valid(data->isValid);

        // std::cout << "track list size:" << data->trackList.size() << std::endl;
        
        for (auto &it: data->trackList) {
            if(_radar_name == "radarfront" && it.id == 0xff) {  // radar front data is valid
                continue;
            }
            else if(_radar_name != "radarfront" && it.id == 0) {  // radar corner data is valid
                continue;
            }
            auto frame = radar_proto->add_track_list();
            frame->set_id(it.id);
            frame->set_track_age(it.trackAge);
            frame->set_obj_obstacle_prob(it.objObstacleProb);
            frame->set_measstate(it.measState);

            frame->mutable_size_lwh()->set_x(static_cast<double>(it.sizeLWH.x));  // float -> double
            frame->mutable_size_lwh()->set_y(static_cast<double>(it.sizeLWH.y));
            frame->mutable_size_lwh()->set_z(static_cast<double>(it.sizeLWH.z));
            // std::cout << "sizeLWH.x" << it.sizeLWH.x   << "->"<< frame->size_lwh().x() << std::endl;
            // std::cout << "sizeLWH.y" << it.sizeLWH.y << "->"<< frame->size_lwh().y() << std::endl;
            // std::cout << "sizeLWH.z" << it.sizeLWH.z << "->"<< frame->size_lwh().z() << std::endl;

            frame->set_orient_agl(static_cast<float>(it.orientAgl));

            frame->mutable_position()->set_x(it.position.x);
            frame->mutable_position()->set_y(it.position.y);
            frame->mutable_position()->set_z(it.position.z);
            frame->mutable_position()->mutable_rms()->set_x(static_cast<double>(it.position.rms.x));
            frame->mutable_position()->mutable_rms()->set_y(static_cast<double>(it.position.rms.y));
            frame->mutable_position()->mutable_rms()->set_z(static_cast<double>(it.position.rms.z));
            frame->mutable_position()->mutable_quality()->set_x(static_cast<double>(it.position.quality.x));
            frame->mutable_position()->mutable_quality()->set_y(static_cast<double>(it.position.quality.y));
            frame->mutable_position()->mutable_quality()->set_z(static_cast<double>(it.position.quality.z));

            frame->mutable_velocity()->set_x(it.velocity.x);
            frame->mutable_velocity()->set_y(it.velocity.y);
            frame->mutable_velocity()->set_z(it.velocity.z);
            frame->mutable_velocity()->mutable_rms()->set_x(static_cast<double>(it.velocity.rms.x));
            frame->mutable_velocity()->mutable_rms()->set_y(static_cast<double>(it.velocity.rms.y));
            frame->mutable_velocity()->mutable_rms()->set_z(static_cast<double>(it.velocity.rms.z));
            frame->mutable_velocity()->mutable_quality()->set_x(static_cast<double>(it.velocity.quality.x));
            frame->mutable_velocity()->mutable_quality()->set_y(static_cast<double>(it.velocity.quality.y));
            frame->mutable_velocity()->mutable_quality()->set_z(static_cast<double>(it.velocity.quality.z));

            frame->mutable_acceleration()->set_x(it.acceleration.x);
            frame->mutable_acceleration()->set_y(it.acceleration.y);
            frame->mutable_acceleration()->set_z(it.acceleration.z);
            frame->mutable_acceleration()->mutable_rms()->set_x(static_cast<double>(it.acceleration.rms.x));
            frame->mutable_acceleration()->mutable_rms()->set_y(static_cast<double>(it.acceleration.rms.y));
            frame->mutable_acceleration()->mutable_rms()->set_z(static_cast<double>(it.acceleration.rms.z));
            frame->mutable_acceleration()->mutable_quality()->set_x(static_cast<double>(it.acceleration.quality.x));
            frame->mutable_acceleration()->mutable_quality()->set_y(static_cast<double>(it.acceleration.quality.y));
            frame->mutable_acceleration()->mutable_quality()->set_z(static_cast<double>(it.acceleration.quality.z));

            frame->set_rcs(static_cast<float>(it.rcs));

            frame->set_snr(static_cast<float>(it.snr));  // double -> float
            // std::cout << "snr:" << it.snr << "->"  << frame->snr() << std::endl;

            frame->set_exist_probability(static_cast<float>(it.existProbability));
            frame->set_mov_property(it.movProperty);
            frame->set_track_type(it.trackType);
        };
        if(_radar_pub_last_seq && (radar_proto->mutable_header()->seq() > _radar_pub_last_seq) 
            && ((radar_proto->mutable_header()->seq() - _radar_pub_last_seq) != 1)) {
            SENSOR_LOG_WARN << "chassis info loss data. receive seq: " << radar_proto->mutable_header()->seq() \
                << " last seq : "  << _radar_pub_last_seq  \
                << " seq diff : " << (radar_proto->mutable_header()->seq() - _radar_pub_last_seq)
                << " interval : " << (radar_proto->mutable_header()->publish_stamp() \
                    - radar_proto->mutable_header()->mutable_sensor_stamp()->radar_stamp()) << " s";
        } else if ((radar_proto->mutable_header()->publish_stamp() 
                - radar_proto->mutable_header()->mutable_sensor_stamp()->radar_stamp()) > 0.05f) {  // 50 ms
            SENSOR_LOG_WARN << "chassis info link latency : " << (radar_proto->mutable_header()->publish_stamp() \
                    - radar_proto->mutable_header()->mutable_sensor_stamp()->radar_stamp()) << " s";
        }
        _radar_pub_last_seq = radar_proto->mutable_header()->seq();

        SetCfgParam(data);
        return radar_proto;
    }

   private:
        template <typename T>
        int SetCfgParam(ara::com::SamplePtr<T> data) {
            uint8_t work_status;
            switch (data->radarState) {
                case 0:
                    work_status = 1;
                    break;
                case 1:
                    work_status = 2;
                    break;
                default:
                    work_status = 0;
                    break;
            }
            if(_radar_status != work_status) {
                int ret = -1;
                if (_radar_name == "radarfront") {
                    ret = ParamConfig::GetInstance().SetParam("system/front_radar_status", work_status);
                }
                else if (_radar_name == "radarcorner1") {
                    ret = ParamConfig::GetInstance().SetParam("system/fr_radar_status", work_status);
                }
                else if (_radar_name == "radarcorner2") {
                    ret = ParamConfig::GetInstance().SetParam("system/fl_radar_status", work_status);
                }
                else if (_radar_name == "radarcorner3") {
                    ret = ParamConfig::GetInstance().SetParam("system/rr_radar_status", work_status);
                }
                else if (_radar_name == "radarcorner4") {
                    ret =ParamConfig::GetInstance().SetParam("system/rl_radar_status", work_status);
                
                }
                if(0 == ret) {
                    _radar_status = work_status;
                }
            } 
            return 0;   
        }

        uint64_t radar_pub_last_time;
        std::string _radar_name;
        uint8_t _radar_status;
        uint8_t _radar_lost_data_fualt_object;
        uint8_t _radar_latency_fualt_object;
        uint8_t _radar_allocat_get_fail_fualt_object;
        uint8_t _radar_alive_monitor_checkpoint_id;
        uint32_t _radar_pub_last_seq;
};

}  // namespace sensor
}  // namespace netaos
}  // namespace hozon