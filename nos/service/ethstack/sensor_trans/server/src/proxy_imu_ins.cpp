#include "proxy_imu_ins.h"
#include "ara/com/sample_ptr.h"
#include "logger.h"
#include "common.h"
#include <cstddef>
#include <memory>


namespace hozon {
namespace netaos {
namespace sensor {
ImuInsProxy::ImuInsProxy () :
imu_seqid(0), 
imu_ins_pub_last_time(0lu),
_imuins_pub_last_seq(0) {

}

std::shared_ptr<hozon::soc::ImuIns> ImuInsProxy::Trans(
        ara::com::SamplePtr<::hozon::netaos::AlgImuInsInfo const> imuIns_info)   {
    // Allocate event memory.
    std::shared_ptr<hozon::soc::ImuIns> imu_data =
                std::make_shared<hozon::soc::ImuIns>();

    if (imu_data == nullptr) {
        SENSOR_LOG_ERROR << "imu_ins_proto Allocat got nullptr!";
        return nullptr;
    }
    imu_data->mutable_header()->set_publish_stamp(GetRealTimestamp());
    imu_data->mutable_header()->set_frame_id(std::string(imuIns_info->header.frameId.begin(),imuIns_info->header.frameId.end()));
    imu_data->mutable_header()->set_seq(imuIns_info->header.seq);
    
    imu_data->mutable_header()->mutable_sensor_stamp()->set_imuins_stamp(
            HafTimeConverStamp(imuIns_info->header.stamp));
    // SENSOR_LOG_DEBUG << "imu_data ->sequence_num: " << imu_seqid;
    imu_data->mutable_header()->set_gnss_stamp(HafTimeConverStamp(imuIns_info->header.gnssStamp));  
    imu_data->set_gps_week(imuIns_info->gpsWeek);
    imu_data->set_gps_sec(imuIns_info->gpsSec);
    
    imu_data->mutable_imu_info()->mutable_angular_velocity()->set_x(
        imuIns_info->imu_info.angularVelocity.x);
    imu_data->mutable_imu_info()->mutable_angular_velocity()->set_y(
        imuIns_info->imu_info.angularVelocity.y);
    imu_data->mutable_imu_info()->mutable_angular_velocity()->set_z(
        imuIns_info->imu_info.angularVelocity.z);
    
    imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_x(
        imuIns_info->imu_info.linearAcceleration.x);
    imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_y(
        imuIns_info->imu_info.linearAcceleration.y);
    imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_z(
        imuIns_info->imu_info.linearAcceleration.z);

    
    imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_x(
        imuIns_info->imu_info.imuVBAngularVelocity.x);
    imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_y(
        imuIns_info->imu_info.imuVBAngularVelocity.y);
    imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_z(
        imuIns_info->imu_info.imuVBAngularVelocity.z);

    
    imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_x(
        imuIns_info->imu_info.imuVBLinearAcceleration.x);
    imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_y(
        imuIns_info->imu_info.imuVBLinearAcceleration.y);   
    imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_z(
        imuIns_info->imu_info.imuVBLinearAcceleration.z);
    
    imu_data->mutable_imu_info()->mutable_gyro_offset()->set_x(
        imuIns_info->imu_info.gyroOffset.x);
    imu_data->mutable_imu_info()->mutable_gyro_offset()->set_y(
        imuIns_info->imu_info.gyroOffset.y);
    imu_data->mutable_imu_info()->mutable_gyro_offset()->set_z(
        imuIns_info->imu_info.gyroOffset.z);
    
    imu_data->mutable_imu_info()->mutable_accel_offset()->set_x(
        imuIns_info->imu_info.accelOffset.x);
    imu_data->mutable_imu_info()->mutable_accel_offset()->set_y(
        imuIns_info->imu_info.accelOffset.y);
    imu_data->mutable_imu_info()->mutable_accel_offset()->set_z(
        imuIns_info->imu_info.accelOffset.z);
    
    imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_x(
        imuIns_info->imu_info.ins2antoffset.x);
    imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_y(
        imuIns_info->imu_info.ins2antoffset.y);
    imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_z(
        imuIns_info->imu_info.ins2antoffset.z);

    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_x(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.x);
    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_y(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.y);
    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_z(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.z);

    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_x(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.x);
    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_y(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.y);
    imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_z(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.z);     

    imu_data->mutable_imu_info()->set_imu_status(imuIns_info->imu_info.imuStatus);
    imu_data->mutable_imu_info()->set_temperature(imuIns_info->imu_info.temperature);
    imu_data->mutable_imu_info()->set_imuyaw(imuIns_info->imu_info.imuyaw);
    
    imu_data->mutable_ins_info()->set_latitude(imuIns_info->ins_info.latitude);
    imu_data->mutable_ins_info()->set_longitude( imuIns_info->ins_info.longitude);
    imu_data->mutable_ins_info()->set_altitude(imuIns_info->ins_info.altitude);

    
    imu_data->mutable_ins_info()->mutable_attitude()->set_x(
        imuIns_info->ins_info.attitude.x);
    imu_data->mutable_ins_info()->mutable_attitude()->set_y(
        imuIns_info->ins_info.attitude.y);
    imu_data->mutable_ins_info()->mutable_attitude()->set_z(
        imuIns_info->ins_info.attitude.z);
    
    imu_data->mutable_ins_info()->mutable_linear_velocity()->set_x(
        imuIns_info->ins_info.linearVelocity.x);
    imu_data->mutable_ins_info()->mutable_linear_velocity()->set_y(
        imuIns_info->ins_info.linearVelocity.y);
    imu_data->mutable_ins_info()->mutable_linear_velocity()->set_z(
        imuIns_info->ins_info.linearVelocity.z);
    
    imu_data->mutable_ins_info()->mutable_augular_velocity()->set_x(
        imuIns_info->ins_info.augularVelocity.x);
    imu_data->mutable_ins_info()->mutable_augular_velocity()->set_y(
        imuIns_info->ins_info.augularVelocity.y);
    imu_data->mutable_ins_info()->mutable_augular_velocity()->set_z(
        imuIns_info->ins_info.augularVelocity.z);
    
    imu_data->mutable_ins_info()->mutable_linear_acceleration()->set_x(
        imuIns_info->ins_info.linearAcceleration.x);
    imu_data->mutable_ins_info()->mutable_linear_acceleration()->set_y(
        imuIns_info->ins_info.linearAcceleration.y);
    imu_data->mutable_ins_info()->mutable_linear_acceleration()->set_z(
        imuIns_info->ins_info.linearAcceleration.z);

    imu_data->mutable_ins_info()->set_heading(imuIns_info->ins_info.heading);

    
    imu_data->mutable_ins_info()->mutable_mounting_error()->set_x(
        imuIns_info->ins_info.mountingError.x);
    imu_data->mutable_ins_info()->mutable_mounting_error()->set_y(
        imuIns_info->ins_info.mountingError.y);
    imu_data->mutable_ins_info()->mutable_mounting_error()->set_z(
        imuIns_info->ins_info.mountingError.z);
    
    imu_data->mutable_ins_info()->mutable_sd_position()->set_x(
        imuIns_info->ins_info.sdPosition.x);
    imu_data->mutable_ins_info()->mutable_sd_position()->set_y(
        imuIns_info->ins_info.sdPosition.y);
    imu_data->mutable_ins_info()->mutable_sd_position()->set_z(
        imuIns_info->ins_info.sdPosition.z);
    
    imu_data->mutable_ins_info()->mutable_sd_attitude()->set_x(
        imuIns_info->ins_info.sdAttitude.x);
    imu_data->mutable_ins_info()->mutable_sd_attitude()->set_y(
        imuIns_info->ins_info.sdAttitude.y);
    imu_data->mutable_ins_info()->mutable_sd_attitude()->set_z(
        imuIns_info->ins_info.sdAttitude.z);
    
    imu_data->mutable_ins_info()->mutable_sd_velocity()->set_x(
        imuIns_info->ins_info.sdVelocity.x);
    imu_data->mutable_ins_info()->mutable_sd_velocity()->set_y(
        imuIns_info->ins_info.sdVelocity.y);
    imu_data->mutable_ins_info()->mutable_sd_velocity()->set_z(
        imuIns_info->ins_info.sdVelocity.z);

    imu_data->mutable_ins_info()->set_sys_status(imuIns_info->ins_info.sysStatus);
    imu_data->mutable_ins_info()->set_gps_status(imuIns_info->ins_info.gpsStatus);
    imu_data->mutable_ins_info()->set_sensor_used(imuIns_info->ins_info.sensorUsed);
    imu_data->mutable_ins_info()->set_wheel_velocity(imuIns_info->ins_info.wheelVelocity);
    imu_data->mutable_ins_info()->set_odo_sf(imuIns_info->ins_info.odoSF);


    imu_data->mutable_offset_info()->mutable_gyo_bias()->set_x(
        imuIns_info->offset_info.gyoBias.x);
    imu_data->mutable_offset_info()->mutable_gyo_bias()->set_y(
        imuIns_info->offset_info.gyoBias.y);
    imu_data->mutable_offset_info()->mutable_gyo_bias()->set_z(
        imuIns_info->offset_info.gyoBias.z);

    imu_data->mutable_offset_info()->mutable_gyo_sf()->set_x(
        imuIns_info->offset_info.gyoSF.x);
    imu_data->mutable_offset_info()->mutable_gyo_sf()->set_y(
        imuIns_info->offset_info.gyoSF.y);
    imu_data->mutable_offset_info()->mutable_gyo_sf()->set_z(
        imuIns_info->offset_info.gyoSF.z);

    imu_data->mutable_offset_info()->mutable_acc_bias()->set_x(
        imuIns_info->offset_info.accBias.x);
    imu_data->mutable_offset_info()->mutable_acc_bias()->set_y(
        imuIns_info->offset_info.accBias.y);
    imu_data->mutable_offset_info()->mutable_acc_bias()->set_z(
        imuIns_info->offset_info.accBias.z);

    imu_data->mutable_offset_info()->mutable_acc_sf()->set_x(
        imuIns_info->offset_info.accSF.x);
    imu_data->mutable_offset_info()->mutable_acc_sf()->set_y(
        imuIns_info->offset_info.accSF.y);
    imu_data->mutable_offset_info()->mutable_acc_sf()->set_z(
        imuIns_info->offset_info.accSF.z);

    if(_imuins_pub_last_seq && (imu_data->mutable_header()->seq() > _imuins_pub_last_seq) 
        && ((imu_data->mutable_header()->seq() - _imuins_pub_last_seq) != 1)) {
        SENSOR_LOG_WARN << "imu ins info lost data. receive seq: " << imu_data->mutable_header()->seq() \
            << " last seq : "  << _imuins_pub_last_seq  \
            << " seq diff : " << (imu_data->mutable_header()->seq() - _imuins_pub_last_seq)
            << " interval : " << (imu_data->mutable_header()->publish_stamp() \
                - imu_data->mutable_header()->mutable_sensor_stamp()->imuins_stamp()) << " s";
    } else if ((imu_data->mutable_header()->publish_stamp() 
            - imu_data->mutable_header()->mutable_sensor_stamp()->imuins_stamp()) > 0.01f) {
        SENSOR_LOG_WARN << "imu ins info link latency : " << (imu_data->mutable_header()->publish_stamp() \
                - imu_data->mutable_header()->mutable_sensor_stamp()->imuins_stamp()) << " s";
    }
    _imuins_pub_last_seq = imu_data->mutable_header()->seq();
    if ((!(imu_data->mutable_header()->seq() % 100)
            && (!imu_ins_pub_last_time || imu_data->mutable_header()->seq()))) {
        PrintOriginalData(imuIns_info);
        struct timespec time;
        if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
            SENSOR_LOG_WARN << "clock_gettime fail ";
        }
        
        uint64_t current_time = (uint64_t)time.tv_sec * 1000lu + ((uint64_t)time.tv_nsec)/1000lu/1000lu;
        uint64_t time_diff = 0;
        if(imu_ins_pub_last_time && (((time_diff = current_time - imu_ins_pub_last_time) - 1000lu) > 100lu)) {
            SENSOR_LOG_WARN << "Send imu ins info seq: " << imu_data->mutable_header()->seq() \
                << " ,interval : " << time_diff << " ms";
        }
        else {
            SENSOR_LOG_INFO << "Send imu ins info seq: " << imu_data->mutable_header()->seq() \
                << " ,interval : " << time_diff << " ms";
        }
        imu_ins_pub_last_time = current_time;
    }
    return imu_data;
} 


void ImuInsProxy::PrintOriginalData(ara::com::SamplePtr<hozon::netaos::AlgImuInsInfo const> imuIns_info) {
    std::string frameId = std::string(imuIns_info->header.frameId.begin(),
                                    imuIns_info->header.frameId.end());
    PRINTSENSORDATA(frameId);
    PRINTSENSORDATA(imuIns_info->header.seq);
    PRINTSENSORDATA(imuIns_info->header.stamp.sec);
    PRINTSENSORDATA(imuIns_info->header.stamp.nsec);

    PRINTSENSORDATA(imuIns_info->header.gnssStamp.sec);
    PRINTSENSORDATA(imuIns_info->header.gnssStamp.nsec);

    PRINTSENSORDATA(imuIns_info->gpsWeek);
    PRINTSENSORDATA(imuIns_info->gpsSec);
    
    PRINTSENSORDATA(
        imuIns_info->imu_info.angularVelocity.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.angularVelocity.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.angularVelocity.z);
    
    PRINTSENSORDATA(
        imuIns_info->imu_info.linearAcceleration.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.linearAcceleration.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.linearAcceleration.z);

    
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBAngularVelocity.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBAngularVelocity.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBAngularVelocity.z);

    
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBLinearAcceleration.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBLinearAcceleration.y);   
    PRINTSENSORDATA(
        imuIns_info->imu_info.imuVBLinearAcceleration.z);
    
    PRINTSENSORDATA(
        imuIns_info->imu_info.gyroOffset.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.gyroOffset.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.gyroOffset.z);
    
    PRINTSENSORDATA(
        imuIns_info->imu_info.accelOffset.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.accelOffset.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.accelOffset.z);
    
    PRINTSENSORDATA(
        imuIns_info->imu_info.ins2antoffset.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.ins2antoffset.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.ins2antoffset.z);

    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.imuPosition.z);

    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.x);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.y);
    PRINTSENSORDATA(
        imuIns_info->imu_info.imu2bodyosffet.eulerAngle.z);     

    PRINTSENSORDATA(imuIns_info->imu_info.imuStatus);
    PRINTSENSORDATA(imuIns_info->imu_info.temperature);
    PRINTSENSORDATA(imuIns_info->imu_info.imuyaw);
    
    PRINTSENSORDATA(imuIns_info->ins_info.latitude);
    PRINTSENSORDATA( imuIns_info->ins_info.longitude);
    PRINTSENSORDATA(imuIns_info->ins_info.altitude);

    
    PRINTSENSORDATA(
        imuIns_info->ins_info.attitude.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.attitude.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.attitude.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearVelocity.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearVelocity.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearVelocity.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.augularVelocity.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.augularVelocity.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.augularVelocity.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearAcceleration.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearAcceleration.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.linearAcceleration.z);

    PRINTSENSORDATA(imuIns_info->ins_info.heading);

    
    PRINTSENSORDATA(
        imuIns_info->ins_info.mountingError.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.mountingError.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.mountingError.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdPosition.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdPosition.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdPosition.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdAttitude.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdAttitude.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdAttitude.z);
    
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdVelocity.x);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdVelocity.y);
    PRINTSENSORDATA(
        imuIns_info->ins_info.sdVelocity.z);

    PRINTSENSORDATA(imuIns_info->ins_info.sysStatus);
    PRINTSENSORDATA(imuIns_info->ins_info.gpsStatus);
    PRINTSENSORDATA(imuIns_info->ins_info.sensorUsed);
    PRINTSENSORDATA(imuIns_info->ins_info.wheelVelocity);
    PRINTSENSORDATA(imuIns_info->ins_info.odoSF);


    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoBias.x);
    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoBias.y);
    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoBias.z);

    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoSF.x);
    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoSF.y);
    PRINTSENSORDATA(
        imuIns_info->offset_info.gyoSF.z);

    PRINTSENSORDATA(
        imuIns_info->offset_info.accBias.x);
    PRINTSENSORDATA(
        imuIns_info->offset_info.accBias.y);
    PRINTSENSORDATA(
        imuIns_info->offset_info.accBias.z);

    PRINTSENSORDATA(
        imuIns_info->offset_info.accSF.x);
    PRINTSENSORDATA(
        imuIns_info->offset_info.accSF.y);
    PRINTSENSORDATA(
        imuIns_info->offset_info.accSF.z);
}

}   // namespace sensor
}   // namespace netaos
}   // namespace hozon
