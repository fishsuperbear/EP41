/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: ReliabilityKind.hpp
 */

#ifndef DDS_CORE_POLICY_RELIABILITY_KIND_HPP
#define DDS_CORE_POLICY_RELIABILITY_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
enum class ReliabilityKind : std::uint8_t {
    BEST_EFFORT,
    RELIABLE
};
}
}
}

#endif /* DDS_CORE_POLICY_RELIABILITY_KIND_HPP */

