/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_HZSOCFAULTSERVICEINTERFACE_COMMON_H
#define HOZON_FM_HZSOCFAULTSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/fm/impl_type_hzfaultevent.h"
#include "impl_type_uint8_t.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace fm {
namespace methods {
namespace FaultEventReport {
struct Output {
    ::uint8_t res;

    static bool IsPlane()
    {
        return true;
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
} // namespace FaultEventReport
} // namespace methods

class HzSocFaultServiceInterface {
public:
    constexpr HzSocFaultServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/Service/HzSocFaultServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace fm
} // namespace hozon

#endif // HOZON_FM_HZSOCFAULTSERVICEINTERFACE_COMMON_H
