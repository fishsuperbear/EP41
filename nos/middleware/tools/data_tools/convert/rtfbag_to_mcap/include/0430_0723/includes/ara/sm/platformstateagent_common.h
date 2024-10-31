/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_PLATFORMSTATEAGENT_COMMON_H
#define ARA_SM_PLATFORMSTATEAGENT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/sm/impl_type_platformstatemsg.h"
#include "impl_type_uint8_t.h"
#include "impl_type_uint32vector.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace sm {
namespace methods {
namespace QueryPlatformState {
struct Output {
    ::uint8_t platformState;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(platformState);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(platformState);
    }

    bool operator==(const Output& t) const
    {
       return (platformState == t.platformState);
    }
};
} // namespace QueryPlatformState
namespace RequestPlatformState {
struct Output {
    ::uint8_t result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace RequestPlatformState
} // namespace methods

class PlatformStateAgent {
public:
    constexpr PlatformStateAgent() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/PlatformStateAgentServiceInterface/PlatformStateAgent");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace sm
} // namespace ara

#endif // ARA_SM_PLATFORMSTATEAGENT_COMMON_H
