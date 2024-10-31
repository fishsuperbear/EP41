/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LPM_SOCPOWERSERVICEINTERFACE_COMMON_H
#define HOZON_LPM_SOCPOWERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_int8.h"
#include "hozon/lpm/impl_type_lpmmcurequest.h"
#include "hozon/lpm/impl_type_lpmsocack.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace lpm {
namespace methods {
namespace RequestLowPower {
struct Output {
    ::hozon::lpm::LpmSocAck lowPowerAck;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lowPowerAck);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lowPowerAck);
    }

    bool operator==(const Output& t) const
    {
       return (lowPowerAck == t.lowPowerAck);
    }
};
} // namespace RequestLowPower
} // namespace methods

class SocPowerServiceInterface {
public:
    constexpr SocPowerServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/SocPowerServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace lpm
} // namespace hozon

#endif // HOZON_LPM_SOCPOWERSERVICEINTERFACE_COMMON_H
