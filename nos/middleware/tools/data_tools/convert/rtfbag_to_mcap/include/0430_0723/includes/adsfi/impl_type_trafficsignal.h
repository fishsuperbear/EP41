/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_TRAFFICSIGNAL_H
#define ADSFI_IMPL_TYPE_TRAFFICSIGNAL_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_rect3d.h"
#include "impl_type_int32array.h"
#include "impl_type_point.h"
#include "impl_type_boolean.h"
#include "ara/common/impl_type_commontime.h"

namespace adsfi {
struct TrafficSignal {
    ::Double confidence;
    ::UInt32 id;
    ::UInt8 color;
    ::Rect3d rect3d;
    ::Int32Array laneId;
    ::Point position;
    ::UInt8 type;
    ::Double distance;
    ::Boolean flash;
    ::Double timeTrack;
    ::Double timeFlash;
    ::Double timeV2X;
    ::ara::common::CommonTime timeCreation;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(confidence);
        fun(id);
        fun(color);
        fun(rect3d);
        fun(laneId);
        fun(position);
        fun(type);
        fun(distance);
        fun(flash);
        fun(timeTrack);
        fun(timeFlash);
        fun(timeV2X);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(confidence);
        fun(id);
        fun(color);
        fun(rect3d);
        fun(laneId);
        fun(position);
        fun(type);
        fun(distance);
        fun(flash);
        fun(timeTrack);
        fun(timeFlash);
        fun(timeV2X);
        fun(timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("confidence", confidence);
        fun("id", id);
        fun("color", color);
        fun("rect3d", rect3d);
        fun("laneId", laneId);
        fun("position", position);
        fun("type", type);
        fun("distance", distance);
        fun("flash", flash);
        fun("timeTrack", timeTrack);
        fun("timeFlash", timeFlash);
        fun("timeV2X", timeV2X);
        fun("timeCreation", timeCreation);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("confidence", confidence);
        fun("id", id);
        fun("color", color);
        fun("rect3d", rect3d);
        fun("laneId", laneId);
        fun("position", position);
        fun("type", type);
        fun("distance", distance);
        fun("flash", flash);
        fun("timeTrack", timeTrack);
        fun("timeFlash", timeFlash);
        fun("timeV2X", timeV2X);
        fun("timeCreation", timeCreation);
    }

    bool operator==(const ::adsfi::TrafficSignal& t) const
    {
        return (fabs(static_cast<double>(confidence - t.confidence)) < DBL_EPSILON) && (id == t.id) && (color == t.color) && (rect3d == t.rect3d) && (laneId == t.laneId) && (position == t.position) && (type == t.type) && (fabs(static_cast<double>(distance - t.distance)) < DBL_EPSILON) && (flash == t.flash) && (fabs(static_cast<double>(timeTrack - t.timeTrack)) < DBL_EPSILON) && (fabs(static_cast<double>(timeFlash - t.timeFlash)) < DBL_EPSILON) && (fabs(static_cast<double>(timeV2X - t.timeV2X)) < DBL_EPSILON) && (timeCreation == t.timeCreation);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_TRAFFICSIGNAL_H
