/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_COMMON_H
#define HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/soc_mcu/impl_type_struct_soc_mcu_array.h"
#include "hozon/soc_mcu/impl_type_egotrajectoryframe_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_locationframe_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_lanelinearray_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_objectfusionframe_soc_mcu.h"
#include "hozon/freespace/impl_type_freespaceframe.h"
#include "hozon/soc_mcu/impl_type_ussinfo_soc_mcu.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace soc_mcu {
namespace methods {
namespace Testadd {
struct Output {
    ::UInt8 res;

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
} // namespace Testadd
} // namespace methods

class HzSocMcuServerServiceInterface {
public:
    constexpr HzSocMcuServerServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzSocMcuServerServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_COMMON_H
