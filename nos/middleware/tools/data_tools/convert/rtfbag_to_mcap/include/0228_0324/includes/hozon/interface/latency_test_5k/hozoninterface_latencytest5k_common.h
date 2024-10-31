/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LATENCY_TEST_5K_HOZONINTERFACE_LATENCYTEST5K_COMMON_H
#define HOZON_INTERFACE_LATENCY_TEST_5K_HOZONINTERFACE_LATENCYTEST5K_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_standardarray5kb.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace latency_test_5k {

class HozonInterface_LatencyTest5K {
public:
    constexpr HozonInterface_LatencyTest5K() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_LatencyTest5K");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace latency_test_5k
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LATENCY_TEST_5K_HOZONINTERFACE_LATENCYTEST5K_COMMON_H
