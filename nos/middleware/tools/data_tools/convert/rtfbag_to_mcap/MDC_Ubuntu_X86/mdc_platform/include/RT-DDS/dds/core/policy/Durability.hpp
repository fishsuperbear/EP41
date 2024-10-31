/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Durability.hpp
 */

#ifndef DDS_CORE_POLICY_DURABILITY_HPP
#define DDS_CORE_POLICY_DURABILITY_HPP

#include <RT-DDS/dds/core/policy/DurabilityKind.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Specifies whether a dds::pub::DataWriter will store and deliver
 * previously published data samples to late-joining dds::sub::DataReader entities.
 *
 * @details \dref_details_DurabilityQosPolicy
 */
class Durability {
public:
    Durability() = default;

    ~Durability() = default;

    static Durability Volatile() noexcept
    {
        return Durability(DurabilityKind::VOLATILE);
    }

    static Durability TransientLocal() noexcept
    {
        return Durability(DurabilityKind::TRANSIENT_LOCAL);
    }

    /**
     * @brief Sets the Durability kind
     *
     * @details \dref_DurabilityQosPolicy_kind
     */
    void Kind(dds::core::policy::DurabilityKind kind) noexcept
    {
        kind_ = kind;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const dds::core::policy::DurabilityKind &Kind() const noexcept
    {
        return kind_;
    }

private:
    explicit Durability(
        dds::core::policy::DurabilityKind kind) noexcept
        : kind_(kind)
    {}

    DurabilityKind kind_;
};
}
}
}

#endif /* DDS_CORE_POLICY_DURABILITY_HPP */

