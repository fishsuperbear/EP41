/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADSFI_IMPL_TYPE_LANELINE_H
#define ARA_ADSFI_IMPL_TYPE_LANELINE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "ara/adsfi/impl_type_laneheader.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_float.h"
#include "ara/adsfi/impl_type_lanelinefitparam.h"
#include "impl_type_pointxy.h"
#include "impl_type_pointxyarray.h"

namespace ara {
namespace adsfi {
struct LaneLine {
    ::Int32 id;
    ::ara::adsfi::LaneHeader header;
    ::UInt8 classification;
    ::String classification_description;
    ::Float classification_confidence;
    ::ara::adsfi::LaneLineFitParam lane_image;
    ::Pointxy start_point;
    ::Pointxy end_point;
    ::PointxyArray point_list;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(header);
        fun(classification);
        fun(classification_description);
        fun(classification_confidence);
        fun(lane_image);
        fun(start_point);
        fun(end_point);
        fun(point_list);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(header);
        fun(classification);
        fun(classification_description);
        fun(classification_confidence);
        fun(lane_image);
        fun(start_point);
        fun(end_point);
        fun(point_list);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("header", header);
        fun("classification", classification);
        fun("classification_description", classification_description);
        fun("classification_confidence", classification_confidence);
        fun("lane_image", lane_image);
        fun("start_point", start_point);
        fun("end_point", end_point);
        fun("point_list", point_list);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("header", header);
        fun("classification", classification);
        fun("classification_description", classification_description);
        fun("classification_confidence", classification_confidence);
        fun("lane_image", lane_image);
        fun("start_point", start_point);
        fun("end_point", end_point);
        fun("point_list", point_list);
    }

    bool operator==(const ::ara::adsfi::LaneLine& t) const
    {
        return (id == t.id) && (header == t.header) && (classification == t.classification) && (classification_description == t.classification_description) && (fabs(static_cast<double>(classification_confidence - t.classification_confidence)) < DBL_EPSILON) && (lane_image == t.lane_image) && (start_point == t.start_point) && (end_point == t.end_point) && (point_list == t.point_list);
    }
};
} // namespace adsfi
} // namespace ara


#endif // ARA_ADSFI_IMPL_TYPE_LANELINE_H
