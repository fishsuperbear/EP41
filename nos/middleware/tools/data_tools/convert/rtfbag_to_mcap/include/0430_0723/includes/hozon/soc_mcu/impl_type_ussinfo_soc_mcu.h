/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_USSINFO_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_USSINFO_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_time.h"
#include "impl_type_uint8.h"
#include "hozon/sensors/impl_type_ussrawdata_pdcifo_avm.h"

namespace hozon {
namespace soc_mcu {
struct UssInfo_soc_mcu {
    ::hozon::soc_mcu::Time time_stamp;
    ::UInt8 counter;
    ::hozon::sensors::UssRawData_PdcIfo_AVM pdcinfo_avm;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(time_stamp);
        fun(counter);
        fun(pdcinfo_avm);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(time_stamp);
        fun(counter);
        fun(pdcinfo_avm);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("time_stamp", time_stamp);
        fun("counter", counter);
        fun("pdcinfo_avm", pdcinfo_avm);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("time_stamp", time_stamp);
        fun("counter", counter);
        fun("pdcinfo_avm", pdcinfo_avm);
    }

    bool operator==(const ::hozon::soc_mcu::UssInfo_soc_mcu& t) const
    {
        return (time_stamp == t.time_stamp) && (counter == t.counter) && (pdcinfo_avm == t.pdcinfo_avm);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_USSINFO_SOC_MCU_H
