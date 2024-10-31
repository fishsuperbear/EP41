/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_COMMON_H
#define HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/lowspeedbsd/impl_type_lowspeedbsdsignal.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace lowspeedbsd {

class HozonInterface_low_spd_bsd {
public:
    constexpr HozonInterface_low_spd_bsd() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_low_spd_bsd");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace lowspeedbsd
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_COMMON_H
