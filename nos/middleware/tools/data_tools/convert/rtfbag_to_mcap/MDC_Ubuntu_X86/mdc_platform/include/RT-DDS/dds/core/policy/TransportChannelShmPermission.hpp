/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportChannelShmPermission.h
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_CHANNEL_SHM_PERMISSION_HPP
#define DDS_CORE_POLICY_TRANSPORT_CHANNEL_SHM_PERMISSION_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
enum class TransportChannelShmPermission : std::uint32_t {
    RWRWOO_MODE = 432U,    /* SHM File Permission 660 */
    RWRWRW_MODE = 438U,    /* SHM File Permission 666 */
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_CHANNEL_SHM_PERMISSION_HPP */

