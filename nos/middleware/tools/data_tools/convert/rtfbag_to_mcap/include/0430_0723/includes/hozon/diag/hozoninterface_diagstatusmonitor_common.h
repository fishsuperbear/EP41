/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_HOZONINTERFACE_DIAGSTATUSMONITOR_COMMON_H
#define HOZON_DIAG_HOZONINTERFACE_DIAGSTATUSMONITOR_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/diag/impl_type_diagstatusframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace diag {

class HozonInterface_DiagStatusMonitor {
public:
    constexpr HozonInterface_DiagStatusMonitor() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_DiagStatusMonitor");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace diag
} // namespace hozon

#endif // HOZON_DIAG_HOZONINTERFACE_DIAGSTATUSMONITOR_COMMON_H