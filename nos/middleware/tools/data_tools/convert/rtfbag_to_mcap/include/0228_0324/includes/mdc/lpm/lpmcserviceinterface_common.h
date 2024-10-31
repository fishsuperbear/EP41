/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_LPM_LPMCSERVICEINTERFACE_COMMON_H
#define MDC_LPM_LPMCSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "mdc/lpm/impl_type_mdcwakeupeventdata.h"
#include "impl_type_int32_t.h"
#include "impl_type_uint32_t.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace lpm {
namespace methods {
namespace SetLowPowerMode {
struct Output {
    ::int32_t errorCode;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(errorCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(errorCode);
    }

    bool operator==(const Output& t) const
    {
       return (errorCode == t.errorCode);
    }
};
} // namespace SetLowPowerMode
namespace SetEnterDeepSleepTime {
struct Output {
    ::int32_t errorCode;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(errorCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(errorCode);
    }

    bool operator==(const Output& t) const
    {
       return (errorCode == t.errorCode);
    }
};
} // namespace SetEnterDeepSleepTime
} // namespace methods

class LpmcServiceInterface {
public:
    constexpr LpmcServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/LpmcServiceInterface/LpmcServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace lpm
} // namespace mdc

#endif // MDC_LPM_LPMCSERVICEINTERFACE_COMMON_H
