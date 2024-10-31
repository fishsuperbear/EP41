/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LATENCY_TEST_2K_HOZONINTERFACE_LATENCYTEST2K_COMMON_H
#define HOZON_INTERFACE_LATENCY_TEST_2K_HOZONINTERFACE_LATENCYTEST2K_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_standardarray2kb.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace latency_test_2k {

class HozonInterface_LatencyTest2K {
public:
    constexpr HozonInterface_LatencyTest2K() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_LatencyTest2K");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace latency_test_2k
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LATENCY_TEST_2K_HOZONINTERFACE_LATENCYTEST2K_COMMON_H
