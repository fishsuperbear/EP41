/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_ABEVENT_H
#define HOZON_STCAMERA_IMPL_TYPE_ABEVENT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace stcamera {
struct ABEvent {
    ::UInt8 event_dist_right;
    ::UInt8 event_dist_left;
    ::UInt8 merge_split_to_track_id_right;
    ::UInt8 merge_split_to_track_id_left;
    ::UInt8 event_quality_right;
    ::UInt8 event_quality_left;
    ::UInt8 event_type_right;
    ::UInt8 event_type_left;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(event_dist_right);
        fun(event_dist_left);
        fun(merge_split_to_track_id_right);
        fun(merge_split_to_track_id_left);
        fun(event_quality_right);
        fun(event_quality_left);
        fun(event_type_right);
        fun(event_type_left);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(event_dist_right);
        fun(event_dist_left);
        fun(merge_split_to_track_id_right);
        fun(merge_split_to_track_id_left);
        fun(event_quality_right);
        fun(event_quality_left);
        fun(event_type_right);
        fun(event_type_left);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("event_dist_right", event_dist_right);
        fun("event_dist_left", event_dist_left);
        fun("merge_split_to_track_id_right", merge_split_to_track_id_right);
        fun("merge_split_to_track_id_left", merge_split_to_track_id_left);
        fun("event_quality_right", event_quality_right);
        fun("event_quality_left", event_quality_left);
        fun("event_type_right", event_type_right);
        fun("event_type_left", event_type_left);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("event_dist_right", event_dist_right);
        fun("event_dist_left", event_dist_left);
        fun("merge_split_to_track_id_right", merge_split_to_track_id_right);
        fun("merge_split_to_track_id_left", merge_split_to_track_id_left);
        fun("event_quality_right", event_quality_right);
        fun("event_quality_left", event_quality_left);
        fun("event_type_right", event_type_right);
        fun("event_type_left", event_type_left);
    }

    bool operator==(const ::hozon::stcamera::ABEvent& t) const
    {
        return (event_dist_right == t.event_dist_right) && (event_dist_left == t.event_dist_left) && (merge_split_to_track_id_right == t.merge_split_to_track_id_right) && (merge_split_to_track_id_left == t.merge_split_to_track_id_left) && (event_quality_right == t.event_quality_right) && (event_quality_left == t.event_quality_left) && (event_type_right == t.event_type_right) && (event_type_left == t.event_type_left);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_ABEVENT_H
