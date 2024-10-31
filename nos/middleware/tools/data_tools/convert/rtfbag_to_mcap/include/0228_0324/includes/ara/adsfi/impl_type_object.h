/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_OBJECT_H
#define ARA_ADSFI_IMPL_TYPE_OBJECT_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_float.h"
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_point.h"
#include "impl_type_rect.h"
#include "impl_type_polygon.h"
#include "impl_type_pointarray.h"
#include "impl_type_twistwithcovariance.h"
#include "impl_type_pathwithcovariancearray.h"
#include "impl_type_int32array.h"
#include "impl_type_uint8array.h"
#include "impl_type_floatarray.h"
#include "impl_type_matrix3d.h"

namespace ara {
namespace adsfi {
struct Object {
    ::ara::common::CommonHeader header;
    ::Float existence_probability;
    ::Int32 id;
    ::ara::common::CommonHeader time_last_update;
    ::ara::common::CommonHeader time_creation;
    ::Int32 measurement_age_count;
    ::UInt8 classification;
    ::String classification_description;
    ::Float classification_confidence;
    ::Float classification_age_seconds;
    ::Int32 classification_age_count;
    ::Point object_box_center;
    ::Point object_box_center_absolute;
    ::Point object_box_center_covariance;
    ::Point object_box_size;
    ::Point object_box_size_covariance;
    ::Rect box_image;
    ::Float object_box_orientation;
    ::Float object_box_orientation_absolute;
    ::Float object_box_orientation_covariance;
    ::Polygon box_polygon;
    ::PointArray box_polygon_absolute;
    ::Point reference_point;
    ::Point reference_point_covariance;
    ::TwistWithCovariance relative_velocity;
    ::TwistWithCovariance absolute_velocity;
    ::TwistWithCovariance enu_velocity;
    ::PointArray sl_velocity;
    ::PointArray contour_points_absolute;
    ::Polygon contour_points;
    ::PathWithCovarianceArray intention_paths;
    ::Float intention_time_step;
    ::Int32Array lane_id;
    ::Uint8Array lane_type;
    ::PointArray position_in_lane;
    ::FloatArray length_in_lane;
    ::FloatArray width_in_lane;
    ::Int32 road_id;
    ::UInt8 road_type;
    ::UInt8 types;
    ::UInt8 cipv_flag;
    ::UInt8 fusion_type;
    ::UInt8 blinker_flag;
    ::Point acceleration;
    ::Point abs_acceleration;
    ::Point anchor_point;
    ::Point abs_anchor_point;
    ::Matrix3d acceleration_covariance;
    ::Matrix3d abs_acceleration_covariance;
    ::Matrix3d velocity_covariance;
    ::Matrix3d abs_velocity_covariance;
    ::UInt8 blinkerStatus;
    ::UInt8 cameraStatus;
    ::UInt8 coordinate;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(existence_probability);
        fun(id);
        fun(time_last_update);
        fun(time_creation);
        fun(measurement_age_count);
        fun(classification);
        fun(classification_description);
        fun(classification_confidence);
        fun(classification_age_seconds);
        fun(classification_age_count);
        fun(object_box_center);
        fun(object_box_center_absolute);
        fun(object_box_center_covariance);
        fun(object_box_size);
        fun(object_box_size_covariance);
        fun(box_image);
        fun(object_box_orientation);
        fun(object_box_orientation_absolute);
        fun(object_box_orientation_covariance);
        fun(box_polygon);
        fun(box_polygon_absolute);
        fun(reference_point);
        fun(reference_point_covariance);
        fun(relative_velocity);
        fun(absolute_velocity);
        fun(enu_velocity);
        fun(sl_velocity);
        fun(contour_points_absolute);
        fun(contour_points);
        fun(intention_paths);
        fun(intention_time_step);
        fun(lane_id);
        fun(lane_type);
        fun(position_in_lane);
        fun(length_in_lane);
        fun(width_in_lane);
        fun(road_id);
        fun(road_type);
        fun(types);
        fun(cipv_flag);
        fun(fusion_type);
        fun(blinker_flag);
        fun(acceleration);
        fun(abs_acceleration);
        fun(anchor_point);
        fun(abs_anchor_point);
        fun(acceleration_covariance);
        fun(abs_acceleration_covariance);
        fun(velocity_covariance);
        fun(abs_velocity_covariance);
        fun(blinkerStatus);
        fun(cameraStatus);
        fun(coordinate);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(existence_probability);
        fun(id);
        fun(time_last_update);
        fun(time_creation);
        fun(measurement_age_count);
        fun(classification);
        fun(classification_description);
        fun(classification_confidence);
        fun(classification_age_seconds);
        fun(classification_age_count);
        fun(object_box_center);
        fun(object_box_center_absolute);
        fun(object_box_center_covariance);
        fun(object_box_size);
        fun(object_box_size_covariance);
        fun(box_image);
        fun(object_box_orientation);
        fun(object_box_orientation_absolute);
        fun(object_box_orientation_covariance);
        fun(box_polygon);
        fun(box_polygon_absolute);
        fun(reference_point);
        fun(reference_point_covariance);
        fun(relative_velocity);
        fun(absolute_velocity);
        fun(enu_velocity);
        fun(sl_velocity);
        fun(contour_points_absolute);
        fun(contour_points);
        fun(intention_paths);
        fun(intention_time_step);
        fun(lane_id);
        fun(lane_type);
        fun(position_in_lane);
        fun(length_in_lane);
        fun(width_in_lane);
        fun(road_id);
        fun(road_type);
        fun(types);
        fun(cipv_flag);
        fun(fusion_type);
        fun(blinker_flag);
        fun(acceleration);
        fun(abs_acceleration);
        fun(anchor_point);
        fun(abs_anchor_point);
        fun(acceleration_covariance);
        fun(abs_acceleration_covariance);
        fun(velocity_covariance);
        fun(abs_velocity_covariance);
        fun(blinkerStatus);
        fun(cameraStatus);
        fun(coordinate);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("existence_probability", existence_probability);
        fun("id", id);
        fun("time_last_update", time_last_update);
        fun("time_creation", time_creation);
        fun("measurement_age_count", measurement_age_count);
        fun("classification", classification);
        fun("classification_description", classification_description);
        fun("classification_confidence", classification_confidence);
        fun("classification_age_seconds", classification_age_seconds);
        fun("classification_age_count", classification_age_count);
        fun("object_box_center", object_box_center);
        fun("object_box_center_absolute", object_box_center_absolute);
        fun("object_box_center_covariance", object_box_center_covariance);
        fun("object_box_size", object_box_size);
        fun("object_box_size_covariance", object_box_size_covariance);
        fun("box_image", box_image);
        fun("object_box_orientation", object_box_orientation);
        fun("object_box_orientation_absolute", object_box_orientation_absolute);
        fun("object_box_orientation_covariance", object_box_orientation_covariance);
        fun("box_polygon", box_polygon);
        fun("box_polygon_absolute", box_polygon_absolute);
        fun("reference_point", reference_point);
        fun("reference_point_covariance", reference_point_covariance);
        fun("relative_velocity", relative_velocity);
        fun("absolute_velocity", absolute_velocity);
        fun("enu_velocity", enu_velocity);
        fun("sl_velocity", sl_velocity);
        fun("contour_points_absolute", contour_points_absolute);
        fun("contour_points", contour_points);
        fun("intention_paths", intention_paths);
        fun("intention_time_step", intention_time_step);
        fun("lane_id", lane_id);
        fun("lane_type", lane_type);
        fun("position_in_lane", position_in_lane);
        fun("length_in_lane", length_in_lane);
        fun("width_in_lane", width_in_lane);
        fun("road_id", road_id);
        fun("road_type", road_type);
        fun("types", types);
        fun("cipv_flag", cipv_flag);
        fun("fusion_type", fusion_type);
        fun("blinker_flag", blinker_flag);
        fun("acceleration", acceleration);
        fun("abs_acceleration", abs_acceleration);
        fun("anchor_point", anchor_point);
        fun("abs_anchor_point", abs_anchor_point);
        fun("acceleration_covariance", acceleration_covariance);
        fun("abs_acceleration_covariance", abs_acceleration_covariance);
        fun("velocity_covariance", velocity_covariance);
        fun("abs_velocity_covariance", abs_velocity_covariance);
        fun("blinkerStatus", blinkerStatus);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("existence_probability", existence_probability);
        fun("id", id);
        fun("time_last_update", time_last_update);
        fun("time_creation", time_creation);
        fun("measurement_age_count", measurement_age_count);
        fun("classification", classification);
        fun("classification_description", classification_description);
        fun("classification_confidence", classification_confidence);
        fun("classification_age_seconds", classification_age_seconds);
        fun("classification_age_count", classification_age_count);
        fun("object_box_center", object_box_center);
        fun("object_box_center_absolute", object_box_center_absolute);
        fun("object_box_center_covariance", object_box_center_covariance);
        fun("object_box_size", object_box_size);
        fun("object_box_size_covariance", object_box_size_covariance);
        fun("box_image", box_image);
        fun("object_box_orientation", object_box_orientation);
        fun("object_box_orientation_absolute", object_box_orientation_absolute);
        fun("object_box_orientation_covariance", object_box_orientation_covariance);
        fun("box_polygon", box_polygon);
        fun("box_polygon_absolute", box_polygon_absolute);
        fun("reference_point", reference_point);
        fun("reference_point_covariance", reference_point_covariance);
        fun("relative_velocity", relative_velocity);
        fun("absolute_velocity", absolute_velocity);
        fun("enu_velocity", enu_velocity);
        fun("sl_velocity", sl_velocity);
        fun("contour_points_absolute", contour_points_absolute);
        fun("contour_points", contour_points);
        fun("intention_paths", intention_paths);
        fun("intention_time_step", intention_time_step);
        fun("lane_id", lane_id);
        fun("lane_type", lane_type);
        fun("position_in_lane", position_in_lane);
        fun("length_in_lane", length_in_lane);
        fun("width_in_lane", width_in_lane);
        fun("road_id", road_id);
        fun("road_type", road_type);
        fun("types", types);
        fun("cipv_flag", cipv_flag);
        fun("fusion_type", fusion_type);
        fun("blinker_flag", blinker_flag);
        fun("acceleration", acceleration);
        fun("abs_acceleration", abs_acceleration);
        fun("anchor_point", anchor_point);
        fun("abs_anchor_point", abs_anchor_point);
        fun("acceleration_covariance", acceleration_covariance);
        fun("abs_acceleration_covariance", abs_acceleration_covariance);
        fun("velocity_covariance", velocity_covariance);
        fun("abs_velocity_covariance", abs_velocity_covariance);
        fun("blinkerStatus", blinkerStatus);
        fun("cameraStatus", cameraStatus);
        fun("coordinate", coordinate);
    }

    bool operator==(const ::ara::adsfi::Object& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(existence_probability - t.existence_probability)) < DBL_EPSILON) && (id == t.id) && (time_last_update == t.time_last_update) && (time_creation == t.time_creation) && (measurement_age_count == t.measurement_age_count) && (classification == t.classification) && (classification_description == t.classification_description) && (fabs(static_cast<double>(classification_confidence - t.classification_confidence)) < DBL_EPSILON) && (fabs(static_cast<double>(classification_age_seconds - t.classification_age_seconds)) < DBL_EPSILON) && (classification_age_count == t.classification_age_count) && (object_box_center == t.object_box_center) && (object_box_center_absolute == t.object_box_center_absolute) && (object_box_center_covariance == t.object_box_center_covariance) && (object_box_size == t.object_box_size) && (object_box_size_covariance == t.object_box_size_covariance) && (box_image == t.box_image) && (fabs(static_cast<double>(object_box_orientation - t.object_box_orientation)) < DBL_EPSILON) && (fabs(static_cast<double>(object_box_orientation_absolute - t.object_box_orientation_absolute)) < DBL_EPSILON) && (fabs(static_cast<double>(object_box_orientation_covariance - t.object_box_orientation_covariance)) < DBL_EPSILON) && (box_polygon == t.box_polygon) && (box_polygon_absolute == t.box_polygon_absolute) && (reference_point == t.reference_point) && (reference_point_covariance == t.reference_point_covariance) && (relative_velocity == t.relative_velocity) && (absolute_velocity == t.absolute_velocity) && (enu_velocity == t.enu_velocity) && (sl_velocity == t.sl_velocity) && (contour_points_absolute == t.contour_points_absolute) && (contour_points == t.contour_points) && (intention_paths == t.intention_paths) && (fabs(static_cast<double>(intention_time_step - t.intention_time_step)) < DBL_EPSILON) && (lane_id == t.lane_id) && (lane_type == t.lane_type) && (position_in_lane == t.position_in_lane) && (length_in_lane == t.length_in_lane) && (width_in_lane == t.width_in_lane) && (road_id == t.road_id) && (road_type == t.road_type) && (types == t.types) && (cipv_flag == t.cipv_flag) && (fusion_type == t.fusion_type) && (blinker_flag == t.blinker_flag) && (acceleration == t.acceleration) && (abs_acceleration == t.abs_acceleration) && (anchor_point == t.anchor_point) && (abs_anchor_point == t.abs_anchor_point) && (acceleration_covariance == t.acceleration_covariance) && (abs_acceleration_covariance == t.abs_acceleration_covariance) && (velocity_covariance == t.velocity_covariance) && (abs_velocity_covariance == t.abs_velocity_covariance) && (blinkerStatus == t.blinkerStatus) && (cameraStatus == t.cameraStatus) && (coordinate == t.coordinate);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_OBJECT_H
