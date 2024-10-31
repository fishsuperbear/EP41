/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_HOZONINTERFACE_DIAGUDS_COMMON_H
#define HOZON_DIAG_HOZONINTERFACE_DIAGUDS_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/diag/impl_type_udsframe.h"
#include "impl_type_int32.h"
#include "hozon/diag/impl_type_mcuudsresframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace diag {
namespace methods {
namespace McuUdsRes {
struct Output {
    ::Int32 res;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(res);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(res);
    }

    bool operator==(const Output& t) const
    {
       return (res == t.res);
    }
};
} // namespace McuUdsRes
} // namespace methods

class HozonInterface_DiagUds {
public:
    constexpr HozonInterface_DiagUds() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_DiagUds");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace diag
} // namespace hozon

#endif // HOZON_DIAG_HOZONINTERFACE_DIAGUDS_COMMON_H
