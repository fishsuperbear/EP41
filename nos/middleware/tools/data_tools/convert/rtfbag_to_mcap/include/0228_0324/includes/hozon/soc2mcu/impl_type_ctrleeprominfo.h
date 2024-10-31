/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_CTRLEEPROMINFO_H
#define HOZON_SOC2MCU_IMPL_TYPE_CTRLEEPROMINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace hozon {
namespace soc2mcu {
struct CtrlEEPromInfo {
    ::Float CtrlYawrateOffset;
    ::Float CtrlYawOffset ;
    ::Float CtrlAxOffset ;
    ::Float CtrlSteerOffset ;
    ::Float CtrlAccelDeadzone;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(CtrlYawrateOffset);
        fun(CtrlYawOffset );
        fun(CtrlAxOffset );
        fun(CtrlSteerOffset );
        fun(CtrlAccelDeadzone);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(CtrlYawrateOffset);
        fun(CtrlYawOffset );
        fun(CtrlAxOffset );
        fun(CtrlSteerOffset );
        fun(CtrlAccelDeadzone);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("CtrlYawrateOffset", CtrlYawrateOffset);
        fun("CtrlYawOffset ", CtrlYawOffset );
        fun("CtrlAxOffset ", CtrlAxOffset );
        fun("CtrlSteerOffset ", CtrlSteerOffset );
        fun("CtrlAccelDeadzone", CtrlAccelDeadzone);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("CtrlYawrateOffset", CtrlYawrateOffset);
        fun("CtrlYawOffset ", CtrlYawOffset );
        fun("CtrlAxOffset ", CtrlAxOffset );
        fun("CtrlSteerOffset ", CtrlSteerOffset );
        fun("CtrlAccelDeadzone", CtrlAccelDeadzone);
    }

    bool operator==(const ::hozon::soc2mcu::CtrlEEPromInfo& t) const
    {
        return (fabs(static_cast<double>(CtrlYawrateOffset - t.CtrlYawrateOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlYawOffset  - t.CtrlYawOffset )) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlAxOffset  - t.CtrlAxOffset )) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlSteerOffset  - t.CtrlSteerOffset )) < DBL_EPSILON) && (fabs(static_cast<double>(CtrlAccelDeadzone - t.CtrlAccelDeadzone)) < DBL_EPSILON);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_CTRLEEPROMINFO_H
