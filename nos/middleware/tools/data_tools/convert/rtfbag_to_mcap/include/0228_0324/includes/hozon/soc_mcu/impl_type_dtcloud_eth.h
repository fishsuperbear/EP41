/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ETH_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ETH_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_soadata.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hafglobaltime.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_ETH {
    ::hozon::soc_mcu::DtCloud_SOAData SOAData;
    ::hozon::soc_mcu::DtCloud_HafGlobalTime HafGlobalTime;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SOAData);
        fun(HafGlobalTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SOAData);
        fun(HafGlobalTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SOAData", SOAData);
        fun("HafGlobalTime", HafGlobalTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SOAData", SOAData);
        fun("HafGlobalTime", HafGlobalTime);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_ETH& t) const
    {
        return (SOAData == t.SOAData) && (HafGlobalTime == t.HafGlobalTime);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_ETH_H
