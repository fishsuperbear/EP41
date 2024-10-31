/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHT_H
#define ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHT_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "impl_type_rect.h"
#include "impl_type_int32array.h"
#include "impl_type_point.h"

namespace ara {
namespace trafficlight {
struct TrafficLight {
    ::Double confidence;
    ::Int32 objectID;
    ::UInt8 color;
    ::Rect box_image;
    ::Int32Array lanes_id;
    ::Point position;
    ::UInt8 type;
    ::Double distance;
    bool flash;
    ::Double time_track;
    ::Double time_flash;
    ::Double time_v2x;
    ::Double time_creation;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(confidence);
        fun(objectID);
        fun(color);
        fun(box_image);
        fun(lanes_id);
        fun(position);
        fun(type);
        fun(distance);
        fun(flash);
        fun(time_track);
        fun(time_flash);
        fun(time_v2x);
        fun(time_creation);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(confidence);
        fun(objectID);
        fun(color);
        fun(box_image);
        fun(lanes_id);
        fun(position);
        fun(type);
        fun(distance);
        fun(flash);
        fun(time_track);
        fun(time_flash);
        fun(time_v2x);
        fun(time_creation);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("confidence", confidence);
        fun("objectID", objectID);
        fun("color", color);
        fun("box_image", box_image);
        fun("lanes_id", lanes_id);
        fun("position", position);
        fun("type", type);
        fun("distance", distance);
        fun("flash", flash);
        fun("time_track", time_track);
        fun("time_flash", time_flash);
        fun("time_v2x", time_v2x);
        fun("time_creation", time_creation);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("confidence", confidence);
        fun("objectID", objectID);
        fun("color", color);
        fun("box_image", box_image);
        fun("lanes_id", lanes_id);
        fun("position", position);
        fun("type", type);
        fun("distance", distance);
        fun("flash", flash);
        fun("time_track", time_track);
        fun("time_flash", time_flash);
        fun("time_v2x", time_v2x);
        fun("time_creation", time_creation);
    }

    bool operator==(const ::ara::trafficlight::TrafficLight& t) const
    {
        return (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (objectID == t.objectID) && (color == t.color) && (box_image == t.box_image) && (lanes_id == t.lanes_id) && (position == t.position) && (type == t.type) && (fabs(static_cast<double>(distance - t.distance)) < DBL_EPSILON) && (flash == t.flash) && (fabs(static_cast<double>(time_track - t.time_track)) < DBL_EPSILON) && (fabs(static_cast<double>(time_flash - t.time_flash)) < DBL_EPSILON) && (fabs(static_cast<double>(time_v2x - t.time_v2x)) < DBL_EPSILON) && (fabs(static_cast<double>(time_creation - t.time_creation)) < DBL_EPSILON);
    }
};
} // namespace trafficlight
} // namespace ara


#endif // ARA_TRAFFICLIGHT_IMPL_TYPE_TRAFFICLIGHT_H
