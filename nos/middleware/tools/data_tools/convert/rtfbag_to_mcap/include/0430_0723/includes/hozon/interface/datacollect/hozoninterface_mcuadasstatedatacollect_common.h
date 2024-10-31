/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUADASSTATEDATACOLLECT_COMMON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUADASSTATEDATACOLLECT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/soc_mcu/impl_type_dtdebug_adas.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace datacollect {

class HozonInterface_MCUAdasStateDataCollect {
public:
    constexpr HozonInterface_MCUAdasStateDataCollect() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_MCUAdasStateDataCollect");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUADASSTATEDATACOLLECT_COMMON_H
