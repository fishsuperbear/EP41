/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLES_H
#define HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLES_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64_t.h"
#include "impl_type_string.h"
#include "impl_type_uint16_t.h"
#include "hozon/radarf/impl_type_contiradarobstaclearray.h"

namespace hozon {
namespace radarf {
struct ContiRadarObstacles {
    ::uint64_t time_sec;
    ::uint64_t time_usec;
    ::String frame_id;
    ::uint16_t sequence_num;
    ::uint16_t error;
    ::hozon::radarf::ContiRadarObstacleArray obstacles;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time_sec);
        fun(time_usec);
        fun(frame_id);
        fun(sequence_num);
        fun(error);
        fun(obstacles);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time_sec);
        fun(time_usec);
        fun(frame_id);
        fun(sequence_num);
        fun(error);
        fun(obstacles);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time_sec", time_sec);
        fun("time_usec", time_usec);
        fun("frame_id", frame_id);
        fun("sequence_num", sequence_num);
        fun("error", error);
        fun("obstacles", obstacles);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time_sec", time_sec);
        fun("time_usec", time_usec);
        fun("frame_id", frame_id);
        fun("sequence_num", sequence_num);
        fun("error", error);
        fun("obstacles", obstacles);
    }

    bool operator==(const ::hozon::radarf::ContiRadarObstacles& t) const
    {
        return (time_sec == t.time_sec) && (time_usec == t.time_usec) && (frame_id == t.frame_id) && (sequence_num == t.sequence_num) && (error == t.error) && (obstacles == t.obstacles);
    }
};
} // namespace radarf
} // namespace hozon


#endif // HOZON_RADARF_IMPL_TYPE_CONTIRADAROBSTACLES_H
