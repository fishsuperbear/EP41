/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CANTP_IMPL_TYPE_CANTPHEADER_H
#define ARA_CANTP_IMPL_TYPE_CANTPHEADER_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "ara/cantp/impl_type_cantptime.h"
#include "impl_type_string.h"

namespace ara {
namespace cantp {
struct CanTpHeader {
    ::UInt32 seq;
    ::ara::cantp::CanTpTime stamp;
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

    bool operator==(const ::ara::cantp::CanTpHeader& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (frame_id == t.frame_id);
    }
};
} // namespace cantp
} // namespace ara


#endif // ARA_CANTP_IMPL_TYPE_CANTPHEADER_H
