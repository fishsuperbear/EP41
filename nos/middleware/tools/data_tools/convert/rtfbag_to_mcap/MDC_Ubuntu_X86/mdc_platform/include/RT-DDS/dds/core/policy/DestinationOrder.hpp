/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DestinationOrder.hpp
 */

#ifndef DDS_CORE_POLICY_DESTINATION_ORDER_HPP
#define DDS_CORE_POLICY_DESTINATION_ORDER_HPP

#include <RT-DDS/dds/core/policy/DestinationOrderKind.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Controls the logical order of updates to the same instance by a
 * dds::pub::Publisher.
 */
class DestinationOrder {
public:
    DestinationOrder() = default;

    ~DestinationOrder() = default;

    static DestinationOrder ByReceptionTimestamp() noexcept
    {
        return DestinationOrder{DestinationOrderKind::BY_RECEPTION_TIMESTAMP};
    }

    static DestinationOrder BySourceTimestamp() noexcept
    {
        return DestinationOrder{DestinationOrderKind::BY_SOURCE_TIMESTAMP};
    }

    /**
     * @brief Sets the destination order kind.
     *
     * **[default]** dds::core::policy::DestinationOrderKind::BY_RECEPTION_TIMESTAMP
     */
    void Kind(DestinationOrderKind k) noexcept
    {
        kind_ = k;
    }

    /**
     * @brief Getter (see the setter with the same name)
     */
    DestinationOrderKind Kind() const noexcept
    {
        return kind_;
    }

private:
    explicit DestinationOrder(DestinationOrderKind k) noexcept
        : kind_(k)
    {}

    DestinationOrderKind kind_;
};
}
}
}

#endif /* DDS_CORE_POLICY_DESTINATION_ORDER_HPP */

