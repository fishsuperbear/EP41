/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_CHASSISCMDHEADER_H
#define ARA_CHASSIS_IMPL_TYPE_CHASSISCMDHEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/chassis/impl_type_chassiscmdtime.h"
#include "impl_type_string.h"

namespace ara {
namespace chassis {
struct ChassisCmdHeader {
    ::UInt32 seq;
    ::ara::chassis::ChassisCmdTime stamp;
    ::String frame_id;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(stamp);
        fun(frame_id);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(stamp);
        fun(frame_id);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frame_id", frame_id);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("frame_id", frame_id);
    }

    bool operator==(const ::ara::chassis::ChassisCmdHeader& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frame_id == t.frame_id);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_CHASSISCMDHEADER_H
