/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LATENCY_TEST_500B_HOZONINTERFACE_LATENCYTEST500B_COMMON_H
#define HOZON_INTERFACE_LATENCY_TEST_500B_HOZONINTERFACE_LATENCYTEST500B_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_standardarray500b.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace latency_test_500b {

class HozonInterface_LatencyTest500B {
public:
    constexpr HozonInterface_LatencyTest500B() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_LatencyTest500B");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace latency_test_500b
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LATENCY_TEST_500B_HOZONINTERFACE_LATENCYTEST500B_COMMON_H
