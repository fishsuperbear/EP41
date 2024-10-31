#include "ins_pvatb_publisher.h"

// #include <sys_ctr.h>

#include <cmath>
#include <ctime>
#include <memory>

#include "can_parser_ins_pvatb.h"
#include "canbus_monitor.h"
#include "canstack_logger.h"
#include "data_buffer.h"
#include "data_type.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "proto/drivers/sensor_imu_ins.pb.h"

using namespace std;
namespace hozon {
namespace netaos {
namespace ins_pvatb {
const std::string defaultCanName = "can7";
int pvatb_wait_inds[] = {HCFDRAWGNSSPVATB, HCFDRAWIMUVB, HCFDINSPVATB};

InsPvatbPublisher* InsPvatbPublisher::sinstance_ = nullptr;
std::mutex g_inspvatb_pub_mutex;

InsPvatbPublisher* InsPvatbPublisher::Instance() {
    if (nullptr == sinstance_) {
        std::lock_guard<std::mutex> lck(g_inspvatb_pub_mutex);
        if (nullptr == sinstance_) {
            sinstance_ = new InsPvatbPublisher();
        }
    }
    return sinstance_;
}

/*
 * Note:
 *         We have already configured the EM-related configuration in the MMC,
 *         so the pport or rport path here should be consistent with it
 */

InsPvatbPublisher::InsPvatbPublisher()
: imu_ins_pub_last_time(0lu) {
    CAN_LOG_INFO << "InsPvatbPublisher::InsPvatbPublisher begin";
    gnss_heading_seqid = 0;
    // gnss_vel_seqid = 0;
    // gnss_pos_seqid = 0;
    imu_seqid = 0;
    // ins_seqid = 0;
    // 注意创建Skeleton的第1个参数是InstanceSpecifier类型，其值为配置文件network_binding.json中instanceSpecifier属性的值
    /* Modify the Port configuration path,Should be consistent with the MMC configuration*/
    skeleton_imu_ins_ = std::make_shared<cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
    
    skeleton_gnss_info_ = std::make_shared<cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
    
    CAN_LOG_INFO << "InsPvatbPublisher::InsPvatbPublisher end";
}

InsPvatbPublisher::~InsPvatbPublisher() {}

int InsPvatbPublisher::Init() {
    // 发布服务
    CAN_LOG_INFO << "InsPvatbPublisher::Init. Offer service.begin";
    skeleton_imu_ins_->Init(0, "/hozon/imu_ins");
    skeleton_gnss_info_->Init(0, "/hozon/gnss_info");
    CAN_LOG_INFO << "InsPvatbPublisher::Init. Offer service.end";

    return 0;
}

template <typename T1, typename T2>
void InsPvatbPublisher::GeometryPoitTransLate(const T1& fromPoint, T2& endPoint) {
    endPoint.set_x(fromPoint.x);
    endPoint.set_y(fromPoint.y);
    endPoint.set_z(fromPoint.z);
};

hozon::common::Point3D InsPvatbPublisher::GeometryPoitTransLate(const GeometryPoit& fromPoint) {
    hozon::common::Point3D endPoint;
    endPoint.set_x(fromPoint.x);
    endPoint.set_y(fromPoint.y);
    endPoint.set_z(fromPoint.z);
    return endPoint;
}

int InsPvatbPublisher::Stop() {
    CAN_LOG_INFO << "InsPvatbPublisher::Stop. begin.";

    // 停止服务
    if (skeleton_gnss_info_) {
        skeleton_gnss_info_->Deinit();
    }

    if (skeleton_imu_ins_) {
        skeleton_imu_ins_->Deinit();
    }

    CAN_LOG_INFO << "InsPvatbPublisher::Stop. end.";

    return 0;
}
double InsPvatbPublisher::GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

int InsPvatbPublisher::PubData(int i, unsigned char* data, int size) {
    struct timespec time;
    if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        CAN_LOG_WARN << "clock_gettime fail ";
    }
    switch (i) {
        case HCFDRAWGNSSPVATB: {
            if (size != sizeof(hozon::netaos::ins_pvatb::GnssInfo)) {
                CAN_LOG_WARN << "radar parser chassis data size error(" << sizeof(hozon::netaos::ins_pvatb::GnssInfo) << "): " << size;
                return -1;
            }
            // auto gnss_info = std::make_unique<hozon::netaos::ins_pvatb::GnssInfo>();
            // memcpy(reinterpret_cast<unsigned char*>(gnss_info.get()), data, size);
            // // // Allocate event memory. 

            // auto gnss_info_data = skeleton_gnss_info->hozonEvent.Allocate();
            // if (gnss_info_data == nullptr) {
            //     CAN_LOG_ERROR << "gnss_data ->mdcEvent.Allocate() got nullptr!";
            //     return -1;
            // }
            // gnss_info_data->header.stamp.sec = time[0].tv_sec;
            // gnss_info_data->header.stamp.nsec = time[0].tv_nsec;
            // gnss_info_data->header.gnssStamp.sec = time[1].tv_sec;
            // gnss_info_data->header.gnssStamp.nsec = time[1].tv_nsec;
            // gnss_info_data->header.frameId = "gnss_info";
            // gnss_info_data->header.seq = ++gnss_heading_seqid;
            // gnss_info_data->gpsWeek = gnss_info->gpsWeek;
            // gnss_info_data->gpsSec = gnss_info->gpsSec;

            // gnss_info_data->gnss_heading.svs = gnss_info->gnss_heading_info.svs;
            // gnss_info_data->gnss_heading.solnSVs = gnss_info->gnss_heading_info.solnSVs;
            // gnss_info_data->gnss_heading.posType = gnss_info->gnss_heading_info.posType;
            // gnss_info_data->gnss_heading.length = gnss_info->gnss_heading_info.length;
            // gnss_info_data->gnss_heading.heading = gnss_info->gnss_heading_info.heading;
            // gnss_info_data->gnss_heading.pitch = gnss_info->gnss_heading_info.pitch;
            // gnss_info_data->gnss_heading.hdgStd = gnss_info->gnss_heading_info.hdgStd;
            // gnss_info_data->gnss_heading.pitchStd = gnss_info->gnss_heading_info.pitchStd;

            // gnss_info_data->gnss_pos.posType = gnss_info->gnss_pos_info.posType;
            // gnss_info_data->gnss_pos.latitude = gnss_info->gnss_pos_info.latitude;
            // gnss_info_data->gnss_pos.longitude = gnss_info->gnss_pos_info.longitude;
            // gnss_info_data->gnss_pos.undulation = gnss_info->gnss_pos_info.undulation;
            // gnss_info_data->gnss_pos.altitude = gnss_info->gnss_pos_info.altitude;
            // gnss_info_data->gnss_pos.latStd = gnss_info->gnss_pos_info.latStd;
            // gnss_info_data->gnss_pos.lonStd = gnss_info->gnss_pos_info.lonStd;
            // gnss_info_data->gnss_pos.hgtStd = gnss_info->gnss_pos_info.hgtStd;
            // gnss_info_data->gnss_pos.svs = gnss_info->gnss_pos_info.svs;
            // gnss_info_data->gnss_pos.solnSVs = gnss_info->gnss_pos_info.solnSVs;
            // gnss_info_data->gnss_pos.diffAge = gnss_info->gnss_pos_info.diffAge;
            // gnss_info_data->gnss_pos.hdop = gnss_info->gnss_pos_info.hdop;
            // gnss_info_data->gnss_pos.vdop = gnss_info->gnss_pos_info.vdop;
            // gnss_info_data->gnss_pos.pdop = gnss_info->gnss_pos_info.pdop;
            // gnss_info_data->gnss_pos.gdop = gnss_info->gnss_pos_info.gdop;
            // gnss_info_data->gnss_pos.tdop = gnss_info->gnss_pos_info.tdop;

            // gnss_info_data->gnss_vel.solStatus = gnss_info->gnss_vel_info.solStatus;
            // gnss_info_data->gnss_vel.horSpd = gnss_info->gnss_vel_info.horSpd;
            // gnss_info_data->gnss_vel.trkGnd = gnss_info->gnss_vel_info.trkGnd;
            // GeometryPoitTransLate(gnss_info->gnss_vel_info.velocity, gnss_info_data->gnss_vel.velocity);
            // GeometryPoitTransLate(gnss_info->gnss_vel_info.velocityStd, gnss_info_data->gnss_vel.velocityStd);
            // if (!(gnss_info_data->header.seq % 100)) {  //1s 
            //     uint64_t current_time = (uint64_t)time[0].tv_sec * 1000lu + ((uint64_t)time[0].tv_nsec)/1000lu/1000lu;
            //     uint64_t time_diff = 0;
            //     if(gnss_pub_last_time && (((time_diff = current_time - gnss_pub_last_time) - 1000lu) > 100lu)) {
            //         CAN_LOG_WARN << "Send gnss info ok  seq: " << gnss_info_data->header.seq 
            //             << " ,interval : " << time_diff << " ms";
            //     }
            //     else {
            //         CAN_LOG_INFO << "Send gnss info ok  seq: " << gnss_info_data->header.seq 
            //             << " ,interval : " << time_diff << " ms";
            //     }
            //     gnss_pub_last_time = current_time;
            // }
            // skeleton_gnss_info->hozonEvent.Send(std::move(gnss_info_data));
            break;
        }
        case HCFDRAWIMUVB: {
            if (size != sizeof(hozon::netaos::ins_pvatb::ImuInsDataInternal)) {
                CAN_LOG_WARN << "radar parser chassis data size error(" << \
                    sizeof(hozon::netaos::ins_pvatb::ImuInsDataInternal) << "): " << size;
                return -1;
            }
            auto imuIns_info = std::make_unique<hozon::netaos::ins_pvatb::ImuInsDataInternal>();
            memcpy(reinterpret_cast<unsigned char*>(imuIns_info.get()), data, size);

            // Allocate event memory.
            std::shared_ptr<hozon::drivers::imuIns::ImuIns> imu_data =
                     std::make_shared<hozon::drivers::imuIns::ImuIns>();

            // auto imu_data = skeleton_imu->hozonEvent.Allocate();
            if (imu_data == nullptr) {
                CAN_LOG_ERROR << "imu_data ->mdcEvent.Allocate() got nullptr!";
                return -1;
            }
            imu_data->mutable_header()->set_timestamp_sec(GetRealTimestamp());
            imu_data->mutable_header()->set_frame_id("imu");
            imu_data->mutable_header()->set_sequence_num(++imu_seqid);
            CAN_LOG_DEBUG << "imu_data ->sequence_num: " << imu_seqid;
            imu_data->set_gnss_stamp_sec(imuIns_info->gpsSec);  //TBD
            imu_data->set_gps_week(imuIns_info->gpsWeek);
            imu_data->set_gps_sec(imuIns_info->gpsSec);
            
            imu_data->mutable_imu_info()->mutable_angular_velocity()->set_x(
                imuIns_info->imub_info.angularVelocity.x);
            imu_data->mutable_imu_info()->mutable_angular_velocity()->set_y(
                imuIns_info->imub_info.angularVelocity.y);
            imu_data->mutable_imu_info()->mutable_angular_velocity()->set_z(
                imuIns_info->imub_info.angularVelocity.z);
            
            imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_x(
                imuIns_info->imub_info.acceleration.x);
            imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_y(
                imuIns_info->imub_info.acceleration.y);
            imu_data->mutable_imu_info()->mutable_linear_acceleration()->set_z(
                imuIns_info->imub_info.acceleration.z);

            
            imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_x(
                imuIns_info->imub_info.imuVBAngularVelocity.x);
            imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_y(
                imuIns_info->imub_info.imuVBAngularVelocity.y);
            imu_data->mutable_imu_info()->mutable_imuvb_angular_velocity()->set_z(
                imuIns_info->imub_info.imuVBAngularVelocity.z);

            
            imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_x(
                imuIns_info->imub_info.imuVBLinearAcceleration.x);
            imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_y(
                imuIns_info->imub_info.imuVBLinearAcceleration.y);   
            imu_data->mutable_imu_info()->mutable_imuvb_linear_acceleration()->set_z(
                imuIns_info->imub_info.imuVBLinearAcceleration.z);
            
            imu_data->mutable_imu_info()->mutable_gyro_offset()->set_x(
                imuIns_info->imub_info.gyroOffset.x);
            imu_data->mutable_imu_info()->mutable_gyro_offset()->set_y(
                imuIns_info->imub_info.gyroOffset.y);
            imu_data->mutable_imu_info()->mutable_gyro_offset()->set_z(
                imuIns_info->imub_info.gyroOffset.z);
            
            imu_data->mutable_imu_info()->mutable_accel_offset()->set_x(
                imuIns_info->imub_info.accelOffset.x);
            imu_data->mutable_imu_info()->mutable_accel_offset()->set_y(
                imuIns_info->imub_info.accelOffset.y);
            imu_data->mutable_imu_info()->mutable_accel_offset()->set_z(
                imuIns_info->imub_info.accelOffset.z);
            
            imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_x(
                imuIns_info->imub_info.ins2antoffset.x);
            imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_y(
                imuIns_info->imub_info.ins2antoffset.y);
            imu_data->mutable_imu_info()->mutable_ins2antoffset()->set_z(
                imuIns_info->imub_info.ins2antoffset.z);

            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_x(
                imuIns_info->imub_info.imu2bodyosffet.imuPosition.x);
            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_y(
                imuIns_info->imub_info.imu2bodyosffet.imuPosition.y);
            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_imu_position()->set_z(
                imuIns_info->imub_info.imu2bodyosffet.imuPosition.z);

            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_x(
                imuIns_info->imub_info.imu2bodyosffet.eulerAngle.x);
            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_y(
                imuIns_info->imub_info.imu2bodyosffet.eulerAngle.y);
            imu_data->mutable_imu_info()->mutable_imu2bodyosffet()->mutable_euler_angle()->set_z(
                imuIns_info->imub_info.imu2bodyosffet.eulerAngle.z);     

            imu_data->mutable_imu_info()->set_imu_status(imuIns_info->imub_info.imuStatus);
            imu_data->mutable_imu_info()->set_temperature(imuIns_info->imub_info.temperature);
            imu_data->mutable_imu_info()->set_imuyaw(imuIns_info->imub_info.imuyaw);
           
            imu_data->mutable_ins_info()->set_latitude(imuIns_info->ins_info.latitude);
            imu_data->mutable_ins_info()->set_longitude( imuIns_info->ins_info.longitude);
            imu_data->mutable_ins_info()->set_elevation(imuIns_info->ins_info.altitude);

            
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
                imuIns_info->imub_info.gyroOffset.x);
            imu_data->mutable_offset_info()->mutable_gyo_bias()->set_y(
                imuIns_info->imub_info.gyroOffset.y);
            imu_data->mutable_offset_info()->mutable_gyo_bias()->set_z(
                imuIns_info->imub_info.gyroOffset.z);

            imu_data->mutable_offset_info()->mutable_gyo_sf()->set_x(
                imuIns_info->imub_info.gyoSF.x);
            imu_data->mutable_offset_info()->mutable_gyo_sf()->set_y(
                imuIns_info->imub_info.gyoSF.y);
            imu_data->mutable_offset_info()->mutable_gyo_sf()->set_z(
                imuIns_info->imub_info.gyoSF.z);

            imu_data->mutable_offset_info()->mutable_acc_bias()->set_x(
                imuIns_info->imub_info.accelOffset.x);
            imu_data->mutable_offset_info()->mutable_acc_bias()->set_y(
                imuIns_info->imub_info.accelOffset.y);
            imu_data->mutable_offset_info()->mutable_acc_bias()->set_z(
                imuIns_info->imub_info.accelOffset.z);

            imu_data->mutable_offset_info()->mutable_acc_sf()->set_x(
                imuIns_info->imub_info.accSF.x);
            imu_data->mutable_offset_info()->mutable_acc_sf()->set_y(
                imuIns_info->imub_info.accSF.y);
            imu_data->mutable_offset_info()->mutable_acc_sf()->set_z(
                imuIns_info->imub_info.accSF.z);

            if (!(imu_data->mutable_header()->sequence_num() % 100)) {
                uint64_t current_time = (uint64_t)time.tv_sec * 1000lu + ((uint64_t)time.tv_nsec)/1000lu/1000lu;
                uint64_t time_diff = 0;
                if(imu_ins_pub_last_time && (((time_diff = current_time - imu_ins_pub_last_time) - 1000lu) > 100lu)) {
                    CAN_LOG_WARN << "Send imu ins info ok  seq: " << imu_data->mutable_header()->sequence_num() \
                        << " ,interval : " << time_diff << " ms";
                }
                else {
                    CAN_LOG_INFO << "Send imu ins info ok  seq: " << imu_data->mutable_header()->sequence_num() \
                        << " ,interval : " << time_diff << " ms";
                }
                imu_ins_pub_last_time = current_time;
            }
            
            // send out proto imu ins data;
            std::shared_ptr<CmProtoBuf> idl_data = std::make_shared<CmProtoBuf>();
            idl_data->name(imu_data->GetTypeName());
            std::string serialized_data;
            imu_data->SerializeToString(&serialized_data);
            idl_data->str().assign(serialized_data.begin(), serialized_data.end());
            if(0 == skeleton_imu_ins_->Write(idl_data)) {
                CAN_LOG_DEBUG << "IMU INS idl data send successful!";
            }
            else {
                CAN_LOG_ERROR << "Imu ins idl data send failde!";
            }
            break;
        }
    }
    
    return 0;
}
void InsPvatbPublisher::Pub() {}
}  // namespace ins_pvatb
}  // namespace canstack
}  // namespace hozon
