/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_HAFHEADER_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_HAFHEADER_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32_t.h"
#include "impl_type_string.h"
#include "hozon/hmi/impl_type_haftime_struct.h"

namespace hozon {
namespace hmi {
struct HafHeader_Struct {
    ::uint32_t seq;
    ::String frameID;
    ::hozon::hmi::HafTime_Struct timeStamp;
    ::hozon::hmi::HafTime_Struct gnssStamp;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(frameID);
        fun(timeStamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(frameID);
        fun(timeStamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("frameID", frameID);
        fun("timeStamp", timeStamp);
        fun("gnssStamp", gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("frameID", frameID);
        fun("timeStamp", timeStamp);
        fun("gnssStamp", gnssStamp);
    }

    bool operator==(const ::hozon::hmi::HafHeader_Struct& t) const
    {
        return (seq == t.seq) && (frameID == t.frameID) && (timeStamp == t.timeStamp) && (gnssStamp == t.gnssStamp);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_HAFHEADER_STRUCT_H
