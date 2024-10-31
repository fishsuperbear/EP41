#pragma once

#include "adf/include/data_types/common/types.h"
#include "logger.h"
#include "proto/perception/transport_element.pb.h"
#include "ara/core/initialization.h"
#include "adf/include/node_bundle.h"
#include "hozon/netaos/impl_type_haflanedetectionoutarray.h"

namespace hozon {
namespace netaos {
namespace sensor {
class SkeletonLaneDetection {
public:
    SkeletonLaneDetection() = default;
    ~SkeletonLaneDetection() = default;
    int Trans(std::string name, adf::NodeBundle recv_data, hozon::netaos::HafLaneDetectionOutArray &data) {
        adf::BaseDataTypePtr idl_data = recv_data.GetOne(name);
        if (idl_data == nullptr) {
            SENSOR_LOG_WARN << "Fail to get " << name << " data.";
            return -1;
        }

        std::shared_ptr<hozon::perception::TransportElement> Sample = 
            std::static_pointer_cast<hozon::perception::TransportElement>(idl_data->proto_msg);

        data.header.seq = Sample->mutable_header()->seq();
        data.header.stamp.sec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp()) * 1e9 / 1e9;
        data.header.stamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->publish_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        data.header.gnssStamp.sec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp()) * 1e9 / 1e9;
        data.header.gnssStamp.nsec = static_cast<uint64_t>(Sample->mutable_header()->gnss_stamp() * 1e9) - data.header.stamp.sec * 1e9;
        // struct timespec time;
        // if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // struct timespec gnss_time;
        // if (0 != clock_gettime(CLOCK_MONOTONIC_RAW, &gnss_time)) {
        //     INTRA_LOG_WARN << "clock_gettime fail ";
        // }
        // data.header.stamp.sec = time.tv_sec;
        // data.header.stamp.nsec = time.tv_nsec;
        // data.header.gnssStamp.sec = gnss_time.tv_sec;
        // data.header.gnssStamp.nsec = gnss_time.tv_nsec;
        std::string frameid = Sample->mutable_header()->frame_id();
        frameid = frameid.substr(0, 20);
        memset(data.header.frameId.data(), 0, 20);
        memcpy(data.header.frameId.data(), frameid.data(), frameid.size());
        for (int32_t i = 0; i < Sample->lane_size(); i++) {
            data.laneDetectionFrontOut[i] = LaneMarker2StructPb(Sample->lane(i), Sample->mutable_header()->sensor_stamp().camera_stamp());
        }
        return 0;
    }

    HafLaneDetectionOut_A LaneMarker2StructPb(const hozon::perception::LaneInfo& ptr_lane, double camera_stamp) {
        HafLaneDetectionOut_A lane_detect_outs;
        ::hozon::netaos::HafLaneDetectionOut alg_lane;
        alg_lane.timeCreation.sec = static_cast<uint64_t>(camera_stamp) * 1e9 / 1e9;
        alg_lane.timeCreation.nsec = static_cast<uint64_t>(camera_stamp * 1e9) - alg_lane.timeCreation.sec * 1e9;
        alg_lane.lanelineSeq = ptr_lane.track_id();
        alg_lane.geometryConfidence = ptr_lane.confidence();
        alg_lane.cls = ptr_lane.lanetype();
        alg_lane.laneLineWidth = 0;
        alg_lane.keyPointVRF.x = ptr_lane.points(0).x();
        alg_lane.keyPointVRF.y = ptr_lane.points(0).y();
        alg_lane.keyPointVRF.z = ptr_lane.points(0).z();
        alg_lane.laneFit.coefficients.a = ptr_lane.lane_param().cubic_curve_set(0).c0();
        alg_lane.laneFit.coefficients.b = ptr_lane.lane_param().cubic_curve_set(0).c1();
        alg_lane.laneFit.coefficients.d = ptr_lane.lane_param().cubic_curve_set(0).c2();
        alg_lane.laneFit.coefficients.c = ptr_lane.lane_param().cubic_curve_set(0).c3();
        alg_lane.laneFit.coefficients.dev_a = 0;
        alg_lane.laneFit.coefficients.dev_b = 0;
        alg_lane.laneFit.coefficients.dev_c = 0;
        alg_lane.laneFit.coefficients.dev_d = 0;
        alg_lane.laneFit.xStartVRF = ptr_lane.lane_param().cubic_curve_set(0).start_point_x();
        alg_lane.laneFit.xEndVRF = ptr_lane.lane_param().cubic_curve_set(0).end_point_x();
        lane_detect_outs[0] = alg_lane;
        return lane_detect_outs;
    }
};    
}
}
}
