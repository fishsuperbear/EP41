/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADER_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADER_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/soc_mcu/impl_type_uint8array_20.h"
#include "hozon/common/impl_type_commontime.h"

namespace hozon {
namespace soc_mcu {
struct CommonHeader_soc_mcu {
    ::UInt32 seq;
    ::hozon::soc_mcu::uint8Array_20 frameId;
    ::hozon::common::CommonTime stamp;
    ::hozon::common::CommonTime gnssStamp;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(frameId);
        fun(stamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(frameId);
        fun(stamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("frameId", frameId);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("frameId", frameId);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
    }

    bool operator==(const ::hozon::soc_mcu::CommonHeader_soc_mcu& t) const
    {
        return (seq == t.seq) && (frameId == t.frameId) && (stamp == t.stamp) && (gnssStamp == t.gnssStamp);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADER_SOC_MCU_H
