/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_HZSMADAPTORSERVICEINTERFACE_COMMON_H
#define HOZON_SM_HZSMADAPTORSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_string.h"
#include "impl_type_uint8_t.h"
#include "hozon/sm/impl_type_fgstatechange.h"
#include "hozon/sm/impl_type_fgstatechangevector.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace sm {
namespace methods {
namespace FuncGroupStateChange {
struct Output {
    ::uint8_t res;

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
} // namespace FuncGroupStateChange
namespace MultiFuncGroupStateChange {
struct Output {
    ::uint8_t res;

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
} // namespace MultiFuncGroupStateChange
namespace QueryFuncFroupState {
struct Output {
    ::String state;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(state);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(state);
    }

    bool operator==(const Output& t) const
    {
       return (state == t.state);
    }
};
} // namespace QueryFuncFroupState
} // namespace methods

class HzSmAdaptorServiceInterface {
public:
    constexpr HzSmAdaptorServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/Service/Provider/HzSmAdaptorServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace sm
} // namespace hozon

#endif // HOZON_SM_HZSMADAPTORSERVICEINTERFACE_COMMON_H
