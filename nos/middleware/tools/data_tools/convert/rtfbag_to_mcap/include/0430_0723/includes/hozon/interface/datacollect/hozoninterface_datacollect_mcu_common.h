/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_MCU_COMMON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_MCU_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_uint8.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace datacollect {
namespace methods {
namespace CollectTriggerReq {
struct Output {
    ::UInt8 Result;

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
} // namespace CollectTriggerReq
} // namespace methods

class HozonInterface_DataCollect_MCU {
public:
    constexpr HozonInterface_DataCollect_MCU() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_DataCollect_MCU");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_MCU_COMMON_H
