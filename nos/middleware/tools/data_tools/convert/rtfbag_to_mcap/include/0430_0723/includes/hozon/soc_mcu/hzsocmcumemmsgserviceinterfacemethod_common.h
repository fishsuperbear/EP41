/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEMETHOD_COMMON_H
#define HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEMETHOD_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_algmcuegomemmsg.h"
#include "impl_type_uint8_t.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace soc_mcu {
namespace methods {
namespace MemMsgReport {
struct Output {
    ::uint8_t Result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Result);
    }

    bool operator==(const Output& t) const
    {
       return (Result == t.Result);
    }
};
} // namespace MemMsgReport
} // namespace methods

class HzSocMcuMemMsgServiceInterfaceMethod {
public:
    constexpr HzSocMcuMemMsgServiceInterfaceMethod() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzSocMcuMemMsgServiceInterfaceMethod");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEMETHOD_COMMON_H
