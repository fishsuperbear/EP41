/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportInterfaces.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_INTERFACES_HPP
#define DDS_CORE_POLICY_TRANSPORT_INTERFACES_HPP

#include <RT-DDS/dds/core/Types.hpp>

namespace dds {
namespace core {
namespace policy {
class TransportInterfaces {
public:
    void Interfaces(dds::core::StringSeq s) noexcept
    {
        interfaces_ = std::move(s);
    }

    const dds::core::StringSeq &Interfaces() const noexcept
    {
        return interfaces_;
    }

private:
    dds::core::StringSeq interfaces_{};
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_INTERFACES_HPP */

