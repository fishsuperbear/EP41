/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DurabilityKind.hpp
 */

#ifndef DDS_CORE_POLICY_DURABILITY_KIND_HPP
#define DDS_CORE_POLICY_DURABILITY_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief The definition of DurabilityKind.
 */
enum class DurabilityKind : std::uint8_t {
    VOLATILE,
    TRANSIENT_LOCAL
};
}
}
}

#endif /* DDS_CORE_POLICY_DURABILITY_KIND_HPP */

