/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_TCB_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_TCB_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint64.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_OsEE_TCB {
    ::UInt8 status;
    ::UInt64 wait_mask;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(status);
        fun(wait_mask);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(status);
        fun(wait_mask);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("status", status);
        fun("wait_mask", wait_mask);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("status", status);
        fun("wait_mask", wait_mask);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_OsEE_TCB& t) const
    {
        return (status == t.status) && (wait_mask == t.wait_mask);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_OSEE_TCB_H
