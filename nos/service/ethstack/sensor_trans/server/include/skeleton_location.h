#pragma once
#include <ctime>
#include <memory>
#include "adf/include/node_bundle.h"
#include "common.h"
#include "proto/localization/localization.pb.h"
#include "logger.h"
#include "hozon/netaos/impl_type_haflocation.h"

namespace hozon {
namespace netaos {
namespace sensor {
class SkeletonLocation {
public:
    SkeletonLocation() {}
    ~SkeletonLocation() {}
    int Trans(std::string name, adf::NodeBundle cm_data, hozon::netaos::HafLocation& data){
        adf::BaseDataTypePtr idl_data = cm_data.GetOne(name);
        if (idl_data == nullptr) {
            SENSOR_LOG_WARN << "Fail to get " << name << " data.";
            return -1;
        }
        std::shared_ptr<hozon::localization::Localization>  Sample =
            std::static_pointer_cast<hozon::localization::Localization>(idl_data->proto_msg);
        
        data.header.seq = Sample->mutable_header()->seq();
        data.header.stamp.sec = static_cast<uint64_t>(Sample->header().data_stamp()) * 1e9 / 1e9;
        data.header.stamp.nsec = static_cast<uint64_t>(Sample->header().data_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        // data.header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
        // data.header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        
        // struct timespec time;
        // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        struct timespec gnss_time;
        if (0 != clock_gettime(CLOCK_VIRTUAL, &gnss_time)) {
            SENSOR_LOG_WARN << "clock_gettime fail ";
        }
        else {
            data.header.gnssStamp.sec = gnss_time.tv_sec;
            data.header.gnssStamp.nsec = gnss_time.tv_nsec;
        }
        // data.header.stamp.sec = time.tv_sec;
        // data.header.stamp.nsec = time.tv_nsec;
        

        std::string frameid = Sample->mutable_header()->frame_id();
        frameid = frameid.substr(0, stringSize);
        memset(data.header.frameId.data(), 0, stringSize);
        memcpy(data.header.frameId.data(), frameid.data(), frameid.size());
        // data.gpsWeek = static_cast<uint64_t>(Sample->measurement_time()) * 1e9 / 1e9 / (7 * 24 * 3600);
        // data.gpsSec = Sample->measurement_time() - data.gpsWeek * (7 * 24 * 3600);
        data.gpsWeek = Sample->gps_week();
        data.gpsSec = Sample->gps_sec();
        data.rtkStatus = Sample->rtk_status();
        // data.coordinateType = Sample->mutable_msf_status()->gnsspos_position_type();
        data.locationState = Sample->location_state();

        if(Sample->mutable_pose()->mutable_wgs()->has_x()) {
            data.pose.poseWGS.position.x = Sample->mutable_pose()->mutable_wgs()->x();
        }
        else {
            data.pose.poseWGS.position.x = 0;
        }
        if(Sample->mutable_pose()->mutable_wgs()->has_y()) {
            data.pose.poseWGS.position.y = Sample->mutable_pose()->mutable_wgs()->y();
        }
        else {
            data.pose.poseWGS.position.y = 0;
        }
        if(Sample->mutable_pose()->mutable_wgs()->has_z()) {
            data.pose.poseWGS.position.z = Sample->mutable_pose()->mutable_wgs()->z();
        }
        else {
            data.pose.poseWGS.position.z = 0;
        }
        // data.pose.poseWGS.position.x = Sample->mutable_pose()->mutable_wgs()->x();
        // data.pose.poseWGS.position.y = Sample->mutable_pose()->mutable_wgs()->y();
        // data.pose.poseWGS.position.z = Sample->mutable_pose()->mutable_wgs()->z();
        data.pose.poseWGS.rotationVRF.x = 0;
        data.pose.poseWGS.rotationVRF.y = 0;
        data.pose.poseWGS.rotationVRF.z = 0;
        data.pose.poseLOCAL.position.x = Sample->mutable_pose_local()->mutable_position()->x();
        data.pose.poseLOCAL.position.y = Sample->mutable_pose_local()->mutable_position()->y();
        data.pose.poseLOCAL.position.z = Sample->mutable_pose_local()->mutable_position()->z();

        
        // data.pose.poseLOCAL.position.x = Sample->mutable_pose()->mutable_pose_local()->x();
        // data.pose.poseLOCAL.position.y = Sample->mutable_pose()->mutable_pose_local()->y();
        // data.pose.poseLOCAL.position.z = Sample->mutable_pose()->mutable_pose_local()->z();
        
        // data.pose.poseLOCAL.eulerAngle.x = Sample->mutable_pose()->mutable_euler_angle()->x();
        // data.pose.poseLOCAL.eulerAngle.y = Sample->mutable_pose()->mutable_euler_angle()->y();
        // data.pose.poseLOCAL.eulerAngle.z = Sample->mutable_pose()->mutable_euler_angle()->z();
        // data.pose.poseLOCAL.rotationVRF.x = Sample->mutable_pose()->mutable_rotation_vrf()->x();
        // data.pose.poseLOCAL.rotationVRF.y = Sample->mutable_pose()->mutable_rotation_vrf()->y();
        // data.pose.poseLOCAL.rotationVRF.z = Sample->mutable_pose()->mutable_rotation_vrf()->z();
        
        if (Sample->pose_local().position().has_x()) {
            data.pose.poseLOCAL.position.x = Sample->pose_local().position().x();
        }
        else {
            data.pose.poseLOCAL.position.x = 0;
        }
        if (Sample->pose_local().position().has_y()) {
            data.pose.poseLOCAL.position.y = Sample->pose_local().position().y();
        }
        else {
            data.pose.poseLOCAL.position.y = 0;
        }
        // data.pose.poseLOCAL.position.x = Sample->pose_local().position().x();
        // data.pose.poseLOCAL.position.y = Sample->pose_local().position().y();
        data.pose.poseLOCAL.position.z = Sample->pose_local().position().z();
        data.pose.poseLOCAL.quaternion.x = Sample->pose().quaternion().x();
        data.pose.poseLOCAL.quaternion.y = Sample->pose().quaternion().y();
        data.pose.poseLOCAL.quaternion.z = Sample->pose().quaternion().z();
        data.pose.poseLOCAL.quaternion.w = Sample->pose().quaternion().w();
        data.pose.poseLOCAL.eulerAngle.x = Sample->pose().euler_angle().x();
        data.pose.poseLOCAL.eulerAngle.y = Sample->pose().euler_angle().y();
        data.pose.poseLOCAL.eulerAngle.z = Sample->pose().euler_angle().z();
        data.pose.poseLOCAL.rotationVRF.x = Sample->pose().rotation_vrf().x();
        data.pose.poseLOCAL.rotationVRF.y = Sample->pose().rotation_vrf().y();
        data.pose.poseLOCAL.rotationVRF.z = Sample->pose().rotation_vrf().z();
        // data.pose.poseLOCAL.heading = Sample->pose_local().heading();
        if (Sample->mutable_pose()->mutable_gcj02()->has_x()) {
            data.pose.poseGCJ02.position.x = Sample->mutable_pose()->mutable_gcj02()->x();
        }
        else {
            data.pose.poseGCJ02.position.x = 0;
        }
        if (Sample->mutable_pose()->mutable_gcj02()->has_y()) {
            data.pose.poseGCJ02.position.y = Sample->mutable_pose()->mutable_gcj02()->y();
        }
        else {
            data.pose.poseGCJ02.position.y = 0;
        }
        // data.pose.poseGCJ02.position.x = Sample->mutable_pose()->mutable_gcj02()->x();
        // data.pose.poseGCJ02.position.y = Sample->mutable_pose()->mutable_gcj02()->y();
        data.pose.poseGCJ02.position.z = Sample->mutable_pose()->mutable_gcj02()->z();
        data.pose.poseGCJ02.quaternion.x = Sample->mutable_pose()->mutable_quaternion()->x();
        data.pose.poseGCJ02.quaternion.y = Sample->mutable_pose()->mutable_quaternion()->y();
        data.pose.poseGCJ02.quaternion.z = Sample->mutable_pose()->mutable_quaternion()->z();
        data.pose.poseGCJ02.quaternion.w = Sample->mutable_pose()->mutable_quaternion()->w();
        data.pose.poseGCJ02.eulerAngle.x = Sample->mutable_pose()->mutable_euler_angle()->x();
        data.pose.poseGCJ02.eulerAngle.y = Sample->mutable_pose()->mutable_euler_angle()->y();
        data.pose.poseGCJ02.eulerAngle.z = Sample->mutable_pose()->mutable_euler_angle()->z();
        data.pose.poseGCJ02.rotationVRF.x = Sample->mutable_pose()->mutable_rotation_vrf()->x();
        data.pose.poseGCJ02.rotationVRF.y = Sample->mutable_pose()->mutable_rotation_vrf()->y();
        data.pose.poseGCJ02.rotationVRF.z = Sample->mutable_pose()->mutable_rotation_vrf()->z();
        if (Sample->mutable_pose()->mutable_pos_utm_01()->has_x()) {
            data.pose.poseUTM01.position.x = Sample->mutable_pose()->mutable_pos_utm_01()->x();
        }
        else {
            data.pose.poseUTM01.position.x = 0;
        }

        if (Sample->mutable_pose()->mutable_pos_utm_01()->has_y()) {
            data.pose.poseUTM01.position.y = Sample->mutable_pose()->mutable_pos_utm_01()->y();
        }
        else {
            data.pose.poseUTM01.position.y = 0;
        }
        // data.pose.poseUTM01.position.x = Sample->mutable_pose()->mutable_pos_utm_01()->x();
        // data.pose.poseUTM01.position.y = Sample->mutable_pose()->mutable_pos_utm_01()->y();
        data.pose.poseUTM01.position.z = Sample->mutable_pose()->mutable_pos_utm_01()->z();
        if (Sample->mutable_pose()->mutable_pos_utm_02()->has_x()) {
            data.pose.poseUTM02.position.x = Sample->mutable_pose()->mutable_pos_utm_02()->x();
        }
        else {
            data.pose.poseUTM02.position.x = 0;
        }
        if (Sample->mutable_pose()->mutable_pos_utm_02()->has_y()) {
            data.pose.poseUTM02.position.y = Sample->mutable_pose()->mutable_pos_utm_02()->y();
        }
        else {
            data.pose.poseUTM02.position.y = 0;
        }
        // data.pose.poseUTM02.position.x = Sample->mutable_pose()->mutable_pos_utm_02()->x();
        // data.pose.poseUTM02.position.y = Sample->mutable_pose()->mutable_pos_utm_02()->y();
        data.pose.poseUTM02.position.z = Sample->mutable_pose()->mutable_pos_utm_02()->z();
        data.pose.poseUTM02.rotationVRF.x = 0;
        data.pose.poseUTM02.rotationVRF.y = 0;
        data.pose.poseUTM02.rotationVRF.z = 0;
        double heading = (Sample->mutable_pose()->heading());
        data.pose.poseUTM01.heading = heading;
        data.pose.poseUTM02.heading = heading;
        // static constexpr double kRAD2DEG = 180 / M_PI;
        // if (Sample->mutable_pose()->has_heading_gcs()) {
        //   heading = -(Sample->mutable_pose()->heading_gcs() - M_PI / 2) * kRAD2DEG;
        // }
        // data.pose.poseWGS.heading = heading;
        // data.pose.poseGCJ02.heading = heading;
        
        
        if (name == "nnplocation") {  // nnp
            data.velocity.twistVRF.linearVRF.x = Sample->mutable_pose()->mutable_linear_velocity_vrf()->x();
            data.velocity.twistVRF.linearVRF.y = Sample->mutable_pose()->mutable_linear_velocity_vrf()->y();
            data.velocity.twistVRF.linearVRF.z = Sample->mutable_pose()->mutable_linear_velocity_vrf()->z();
            data.velocity.twistVRF.angularVRF.x = Sample->mutable_pose()->mutable_angular_velocity_vrf()->x();
            data.velocity.twistVRF.angularVRF.y = Sample->mutable_pose()->mutable_angular_velocity_vrf()->y();
            data.velocity.twistVRF.angularVRF.z = Sample->mutable_pose()->mutable_angular_velocity_vrf()->z();
            
            data.acceleration.linearVRF.linearVRF.x = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->x();
            data.acceleration.linearVRF.linearVRF.y = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->y();
            data.acceleration.linearVRF.linearVRF.z = Sample->mutable_pose()->mutable_linear_acceleration_vrf()->z();
            data.pose.poseLOCAL.heading = Sample->mutable_pose_local()->local_heading();
        }
        else { // avp
            data.velocity.twistVRF.linearVRF.x = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->x();
            data.velocity.twistVRF.linearVRF.y = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->y();
            data.velocity.twistVRF.linearVRF.z = Sample->mutable_pose_local()->mutable_linear_velocity_vrf()->z();
            data.velocity.twistVRF.angularVRF.x = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->x();
            data.velocity.twistVRF.angularVRF.y = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->y();
            data.velocity.twistVRF.angularVRF.z = Sample->mutable_pose_local()->mutable_angular_velocity_vrf()->z();
            data.acceleration.linearVRF.linearVRF.x = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->x();
            data.acceleration.linearVRF.linearVRF.y = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->y();
            data.acceleration.linearVRF.linearVRF.z = Sample->mutable_pose_local()->mutable_linear_acceleration_vrf()->z();
            data.pose.poseLOCAL.heading = Sample->mutable_pose_local()->heading();
        }

        data.acceleration.linearVRF.angularVRF.x = 0;
        data.acceleration.linearVRF.angularVRF.y = 0;
        data.acceleration.linearVRF.angularVRF.z = 0;
        
        // data.acceleration.linearVRF.angularVRF.x = 0;
        // data.acceleration.linearVRF.angularVRF.y = 0;
        // data.acceleration.linearVRF.angularVRF.z = 0;
        //   data.velocity.twistVRFVRF.linearVRF.x =
        //       Sample->mutable_pose().linear_velocity_vrf().y();
        //   data.velocity.twistVRFVRF.linearVRF.y =
        //       -Sample->mutable_pose().linear_velocity_vrf().x();
        //   data.velocity.twistVRFVRF.linearVRF.z =
        //       Sample->mutable_pose().linear_velocity_vrf().z();
        //   static constexpr double kG = 9.80665;
        //   data.velocity.twistVRFVRF.angularVRF.x =
        //       Sample->mutable_pose().angular_velocity_vrf().y() * kRAD2DEG;
        //   data.velocity.twistVRFVRF.angularVRF.y =
        //       -Sample->mutable_pose().angular_velocity_vrf().x() * kRAD2DEG;
        //   data.velocity.twistVRFVRF.angularVRF.z =
        //       Sample->mutable_pose().angular_velocity_vrf().z() * kRAD2DEG;
        //   data.velocity.twistVRFVRF.angularRawVRF.x =
        //       Sample->mutable_pose().angular_velocity_vrf().y() * kRAD2DEG;
        //   data.velocity.twistVRFVRF.angularRawVRF.y =
        //       -Sample->mutable_pose().angular_velocity_vrf().x() * kRAD2DEG;
        //   data.velocity.twistVRFVRF.angularRawVRF.z =
        //       Sample->mutable_pose().angular_velocity_vrf().z() * kRAD2DEG;
        //   data.acceleration.linearVRF.linearVRF.x =
        //       Sample->mutable_pose().linear_acceleration_vrf().y() / kG;
        //   data.acceleration.linearVRF.linearVRF.y =
        //       -Sample->mutable_pose().linear_acceleration_vrf().x() / kG;
        //   data.acceleration.linearVRF.linearVRF.z =
        //       -Sample->mutable_pose().linear_acceleration_vrf().z() / kG;
        //   data.acceleration.linearVRF.linearRawVRF.x =
        //       Sample->mutable_pose().linear_acceleration_vrf().y() / kG;
        //   data.acceleration.linearVRF.linearRawVRF.y =
        //       -Sample->mutable_pose().linear_acceleration_vrf().x() / kG;
        //   data.acceleration.linearVRF.linearRawVRF.z =
        //       -Sample->mutable_pose().linear_acceleration_vrf().z() / kG;
        data.pose.utmZoneID01 = Sample->mutable_pose()->utm_zone_01();
        data.pose.utmZoneID02 = Sample->mutable_pose()->utm_zone_02();
        PrintSendData(data);
        return 0;
    }
    
    void PrintSendData(hozon::netaos::HafLocation& data) {
        PRINTSENSORDATA(data.header.seq);
        PRINTSENSORDATA(data.header.gnssStamp.nsec);
        PRINTSENSORDATA(data.header.gnssStamp.sec);
        PRINTSENSORDATA(data.header.stamp.nsec);
        PRINTSENSORDATA(data.header.stamp.sec);
    }
};  
}
}
}