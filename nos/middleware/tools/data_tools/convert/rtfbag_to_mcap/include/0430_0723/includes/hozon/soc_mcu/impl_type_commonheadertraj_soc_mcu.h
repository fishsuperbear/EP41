/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADERTRAJ_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADERTRAJ_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/common/impl_type_commontime.h"

namespace hozon {
namespace soc_mcu {
struct CommonHeaderTraj_soc_mcu {
    ::UInt32 seq;
    ::hozon::common::CommonTime stamp;
    ::hozon::common::CommonTime gnssStamp;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(seq);
        fun(stamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(seq);
        fun(stamp);
        fun(gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("seq", seq);
        fun("stamp", stamp);
        fun("gnssStamp", gnssStamp);
    }

    bool operator==(const ::hozon::soc_mcu::CommonHeaderTraj_soc_mcu& t) const
    {
        return (seq == t.seq) && (stamp == t.stamp) && (gnssStamp == t.gnssStamp);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_COMMONHEADERTRAJ_SOC_MCU_H
