/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DestinationOrderKind.hpp
 */

#ifndef DDS_CORE_POLICY_DESTINATION_ORDER_KIND_HPP
#define DDS_CORE_POLICY_DESTINATION_ORDER_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief The definition of DestinationOrderKind.
 */
enum class DestinationOrderKind : std::uint8_t {
    BY_RECEPTION_TIMESTAMP,
    BY_SOURCE_TIMESTAMP
};
}
}
}

#endif /* DDS_CORE_POLICY_DESTINATION_ORDER_KIND_HPP */

