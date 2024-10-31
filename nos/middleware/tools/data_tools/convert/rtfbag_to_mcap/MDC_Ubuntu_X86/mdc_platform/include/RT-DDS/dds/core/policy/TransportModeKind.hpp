/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_MODE_KIND_HPP
#define DDS_CORE_POLICY_TRANSPORT_MODE_KIND_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief The definition of DestinationOrderKind.
 */
enum class TransportModeKind : std::uint8_t {
    TRANSPORT_ASYNCHRONOUS_MODE,
    TRANSPORT_SYNCHRONOUS_MODE
};
}
}
}

#endif // DDS_CORE_POLICY_TRANSPORT_MODE_KIND_HPP
