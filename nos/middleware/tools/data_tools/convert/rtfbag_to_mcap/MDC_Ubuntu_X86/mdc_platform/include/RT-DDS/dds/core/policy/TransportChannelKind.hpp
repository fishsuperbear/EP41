/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportChannelKind.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_CHANNEL_KIND_HPP
#define DDS_CORE_POLICY_TRANSPORT_CHANNEL_KIND_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace core {
namespace policy {
enum class TransportChannelKind : std::uint32_t {
    UDP = 0x0001U,
    SHM = 0x0002U,
    DSHM = 0x0004U,
    ICC = 0x0008U,
    UDP_SHM      = SHM  | UDP,
    UDP_DSHM     = DSHM | UDP,
    UDP_ICC      = ICC  | UDP,
    SHM_ICC      = ICC  | SHM,
    UDP_SHM_ICC  = ICC  | SHM  | UDP,
    DSHM_ICC     = ICC  | DSHM,
    UDP_DSHM_ICC = ICC  | DSHM | UDP,
    UDP2DSHM = 0x0015U,
    ICC2DSHM = 0x0010U | ICC | DSHM, /* 0x1C */
};

/**
 * @brief A helper function, to convert the kind to a readable string
 * @param kind the TransportChannelKind to convert
 * @return a copy of the readable string
 */
std::string TransportChannelKindToStr(TransportChannelKind kind);
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_CHANNEL_KIND_HPP */

