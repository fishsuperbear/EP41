/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_ESTOP_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_ESTOP_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_uint8array_20.h"
#include "hozon/soc_mcu/impl_type_commonheader_soc_mcu.h"

namespace hozon {
namespace soc_mcu {
struct Estop_soc_mcu {
    ::UInt8 isStop;
    ::hozon::soc_mcu::uint8Array_20 description;
    ::hozon::soc_mcu::CommonHeader_soc_mcu header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(isStop);
        fun(description);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("isStop", isStop);
        fun("description", description);
        fun("header", header);
    }

    bool operator==(const ::hozon::soc_mcu::Estop_soc_mcu& t) const
    {
        return (isStop == t.isStop) && (description == t.description) && (header == t.header);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_ESTOP_SOC_MCU_H
