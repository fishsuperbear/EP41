/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2NVMTYPE_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2NVMTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc2mcu {
struct HafAEB2NVMType {
    ::UInt8 ADCS8_FCWSensitiveLevel;
    ::UInt32 ADCS_AEB_Version;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS8_FCWSensitiveLevel);
        fun(ADCS_AEB_Version);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS8_FCWSensitiveLevel);
        fun(ADCS_AEB_Version);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("ADCS_AEB_Version", ADCS_AEB_Version);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("ADCS_AEB_Version", ADCS_AEB_Version);
    }

    bool operator==(const ::hozon::soc2mcu::HafAEB2NVMType& t) const
    {
        return (ADCS8_FCWSensitiveLevel == t.ADCS8_FCWSensitiveLevel) && (ADCS_AEB_Version == t.ADCS_AEB_Version);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2NVMTYPE_H
