/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_CAMOBJECT_H
#define HOZON_STCAMERA_IMPL_TYPE_CAMOBJECT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_double.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace stcamera {
struct CamObject {
    ::UInt8 obj_id;
    ::UInt8 maintenance_state;
    ::Double dyn_confidence;
    ::UInt8 ref_point;
    ::UInt8 prob_of_exist;
    ::UInt8 obj_life_cycles;
    ::UInt8 object_occlusion;
    ::UInt8 motion_status;
    ::Double dist_x;
    ::Double dist_y;
    ::Double vabs_x;
    ::Double vabs_y;
    ::Double aabs_x;
    ::Double aabs_y;
    ::Double length;
    ::UInt8 orientation;
    ::Double width;
    ::Double dist_x_std;
    ::Double dist_y_std;
    ::Double vabs_x_std;
    ::Double vabs_y_std;
    ::Double aabs_x_std;
    ::Double aabs_y_std;
    ::Double height;
    ::Double offset_to_ground;
    ::UInt8 turn_indicator;
    ::UInt8 lane_movement;
    ::UInt8 lane_assignment;
    ::UInt8 rel_width_on_assoc_lane;
    ::UInt8 brake_light;
    ::UInt8 cam_class_conf_highest;
    ::UInt8 cam_class_conf_second;
    ::Double class_conf_conf_highest;
    ::Double class_conf_conf_second;
    ::UInt16 eba_inhibition_mask;
    ::UInt8 eba_obj_quality;
    ::UInt8 acc_obj_quality;
    ::Double corr_pos_x;
    ::Double corr_pos_y;
    ::Double corr_vel_x;
    ::Double corr_vel_y;
    ::Double corr_acc_x;
    ::Double corr_acc_y;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(obj_id);
        fun(maintenance_state);
        fun(dyn_confidence);
        fun(ref_point);
        fun(prob_of_exist);
        fun(obj_life_cycles);
        fun(object_occlusion);
        fun(motion_status);
        fun(dist_x);
        fun(dist_y);
        fun(vabs_x);
        fun(vabs_y);
        fun(aabs_x);
        fun(aabs_y);
        fun(length);
        fun(orientation);
        fun(width);
        fun(dist_x_std);
        fun(dist_y_std);
        fun(vabs_x_std);
        fun(vabs_y_std);
        fun(aabs_x_std);
        fun(aabs_y_std);
        fun(height);
        fun(offset_to_ground);
        fun(turn_indicator);
        fun(lane_movement);
        fun(lane_assignment);
        fun(rel_width_on_assoc_lane);
        fun(brake_light);
        fun(cam_class_conf_highest);
        fun(cam_class_conf_second);
        fun(class_conf_conf_highest);
        fun(class_conf_conf_second);
        fun(eba_inhibition_mask);
        fun(eba_obj_quality);
        fun(acc_obj_quality);
        fun(corr_pos_x);
        fun(corr_pos_y);
        fun(corr_vel_x);
        fun(corr_vel_y);
        fun(corr_acc_x);
        fun(corr_acc_y);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(obj_id);
        fun(maintenance_state);
        fun(dyn_confidence);
        fun(ref_point);
        fun(prob_of_exist);
        fun(obj_life_cycles);
        fun(object_occlusion);
        fun(motion_status);
        fun(dist_x);
        fun(dist_y);
        fun(vabs_x);
        fun(vabs_y);
        fun(aabs_x);
        fun(aabs_y);
        fun(length);
        fun(orientation);
        fun(width);
        fun(dist_x_std);
        fun(dist_y_std);
        fun(vabs_x_std);
        fun(vabs_y_std);
        fun(aabs_x_std);
        fun(aabs_y_std);
        fun(height);
        fun(offset_to_ground);
        fun(turn_indicator);
        fun(lane_movement);
        fun(lane_assignment);
        fun(rel_width_on_assoc_lane);
        fun(brake_light);
        fun(cam_class_conf_highest);
        fun(cam_class_conf_second);
        fun(class_conf_conf_highest);
        fun(class_conf_conf_second);
        fun(eba_inhibition_mask);
        fun(eba_obj_quality);
        fun(acc_obj_quality);
        fun(corr_pos_x);
        fun(corr_pos_y);
        fun(corr_vel_x);
        fun(corr_vel_y);
        fun(corr_acc_x);
        fun(corr_acc_y);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("obj_id", obj_id);
        fun("maintenance_state", maintenance_state);
        fun("dyn_confidence", dyn_confidence);
        fun("ref_point", ref_point);
        fun("prob_of_exist", prob_of_exist);
        fun("obj_life_cycles", obj_life_cycles);
        fun("object_occlusion", object_occlusion);
        fun("motion_status", motion_status);
        fun("dist_x", dist_x);
        fun("dist_y", dist_y);
        fun("vabs_x", vabs_x);
        fun("vabs_y", vabs_y);
        fun("aabs_x", aabs_x);
        fun("aabs_y", aabs_y);
        fun("length", length);
        fun("orientation", orientation);
        fun("width", width);
        fun("dist_x_std", dist_x_std);
        fun("dist_y_std", dist_y_std);
        fun("vabs_x_std", vabs_x_std);
        fun("vabs_y_std", vabs_y_std);
        fun("aabs_x_std", aabs_x_std);
        fun("aabs_y_std", aabs_y_std);
        fun("height", height);
        fun("offset_to_ground", offset_to_ground);
        fun("turn_indicator", turn_indicator);
        fun("lane_movement", lane_movement);
        fun("lane_assignment", lane_assignment);
        fun("rel_width_on_assoc_lane", rel_width_on_assoc_lane);
        fun("brake_light", brake_light);
        fun("cam_class_conf_highest", cam_class_conf_highest);
        fun("cam_class_conf_second", cam_class_conf_second);
        fun("class_conf_conf_highest", class_conf_conf_highest);
        fun("class_conf_conf_second", class_conf_conf_second);
        fun("eba_inhibition_mask", eba_inhibition_mask);
        fun("eba_obj_quality", eba_obj_quality);
        fun("acc_obj_quality", acc_obj_quality);
        fun("corr_pos_x", corr_pos_x);
        fun("corr_pos_y", corr_pos_y);
        fun("corr_vel_x", corr_vel_x);
        fun("corr_vel_y", corr_vel_y);
        fun("corr_acc_x", corr_acc_x);
        fun("corr_acc_y", corr_acc_y);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("obj_id", obj_id);
        fun("maintenance_state", maintenance_state);
        fun("dyn_confidence", dyn_confidence);
        fun("ref_point", ref_point);
        fun("prob_of_exist", prob_of_exist);
        fun("obj_life_cycles", obj_life_cycles);
        fun("object_occlusion", object_occlusion);
        fun("motion_status", motion_status);
        fun("dist_x", dist_x);
        fun("dist_y", dist_y);
        fun("vabs_x", vabs_x);
        fun("vabs_y", vabs_y);
        fun("aabs_x", aabs_x);
        fun("aabs_y", aabs_y);
        fun("length", length);
        fun("orientation", orientation);
        fun("width", width);
        fun("dist_x_std", dist_x_std);
        fun("dist_y_std", dist_y_std);
        fun("vabs_x_std", vabs_x_std);
        fun("vabs_y_std", vabs_y_std);
        fun("aabs_x_std", aabs_x_std);
        fun("aabs_y_std", aabs_y_std);
        fun("height", height);
        fun("offset_to_ground", offset_to_ground);
        fun("turn_indicator", turn_indicator);
        fun("lane_movement", lane_movement);
        fun("lane_assignment", lane_assignment);
        fun("rel_width_on_assoc_lane", rel_width_on_assoc_lane);
        fun("brake_light", brake_light);
        fun("cam_class_conf_highest", cam_class_conf_highest);
        fun("cam_class_conf_second", cam_class_conf_second);
        fun("class_conf_conf_highest", class_conf_conf_highest);
        fun("class_conf_conf_second", class_conf_conf_second);
        fun("eba_inhibition_mask", eba_inhibition_mask);
        fun("eba_obj_quality", eba_obj_quality);
        fun("acc_obj_quality", acc_obj_quality);
        fun("corr_pos_x", corr_pos_x);
        fun("corr_pos_y", corr_pos_y);
        fun("corr_vel_x", corr_vel_x);
        fun("corr_vel_y", corr_vel_y);
        fun("corr_acc_x", corr_acc_x);
        fun("corr_acc_y", corr_acc_y);
    }

    bool operator==(const ::hozon::stcamera::CamObject& t) const
    {
        return (obj_id == t.obj_id) && (maintenance_state == t.maintenance_state) && (fabs(static_cast<double>(dyn_confidence - t.dyn_confidence)) < DBL_EPSILON) && (ref_point == t.ref_point) && (prob_of_exist == t.prob_of_exist) && (obj_life_cycles == t.obj_life_cycles) && (object_occlusion == t.object_occlusion) && (motion_status == t.motion_status) && (fabs(static_cast<double>(dist_x - t.dist_x)) < DBL_EPSILON) && (fabs(static_cast<double>(dist_y - t.dist_y)) < DBL_EPSILON) && (fabs(static_cast<double>(vabs_x - t.vabs_x)) < DBL_EPSILON) && (fabs(static_cast<double>(vabs_y - t.vabs_y)) < DBL_EPSILON) && (fabs(static_cast<double>(aabs_x - t.aabs_x)) < DBL_EPSILON) && (fabs(static_cast<double>(aabs_y - t.aabs_y)) < DBL_EPSILON) && (fabs(static_cast<double>(length - t.length)) < DBL_EPSILON) && (orientation == t.orientation) && (fabs(static_cast<double>(width - t.width)) < DBL_EPSILON) && (fabs(static_cast<double>(dist_x_std - t.dist_x_std)) < DBL_EPSILON) && (fabs(static_cast<double>(dist_y_std - t.dist_y_std)) < DBL_EPSILON) && (fabs(static_cast<double>(vabs_x_std - t.vabs_x_std)) < DBL_EPSILON) && (fabs(static_cast<double>(vabs_y_std - t.vabs_y_std)) < DBL_EPSILON) && (fabs(static_cast<double>(aabs_x_std - t.aabs_x_std)) < DBL_EPSILON) && (fabs(static_cast<double>(aabs_y_std - t.aabs_y_std)) < DBL_EPSILON) && (fabs(static_cast<double>(height - t.height)) < DBL_EPSILON) && (fabs(static_cast<double>(offset_to_ground - t.offset_to_ground)) < DBL_EPSILON) && (turn_indicator == t.turn_indicator) && (lane_movement == t.lane_movement) && (lane_assignment == t.lane_assignment) && (rel_width_on_assoc_lane == t.rel_width_on_assoc_lane) && (brake_light == t.brake_light) && (cam_class_conf_highest == t.cam_class_conf_highest) && (cam_class_conf_second == t.cam_class_conf_second) && (fabs(static_cast<double>(class_conf_conf_highest - t.class_conf_conf_highest)) < DBL_EPSILON) && (fabs(static_cast<double>(class_conf_conf_second - t.class_conf_conf_second)) < DBL_EPSILON) && (eba_inhibition_mask == t.eba_inhibition_mask) && (eba_obj_quality == t.eba_obj_quality) && (acc_obj_quality == t.acc_obj_quality) && (fabs(static_cast<double>(corr_pos_x - t.corr_pos_x)) < DBL_EPSILON) && (fabs(static_cast<double>(corr_pos_y - t.corr_pos_y)) < DBL_EPSILON) && (fabs(static_cast<double>(corr_vel_x - t.corr_vel_x)) < DBL_EPSILON) && (fabs(static_cast<double>(corr_vel_y - t.corr_vel_y)) < DBL_EPSILON) && (fabs(static_cast<double>(corr_acc_x - t.corr_acc_x)) < DBL_EPSILON) && (fabs(static_cast<double>(corr_acc_y - t.corr_acc_y)) < DBL_EPSILON);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_CAMOBJECT_H
