/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: State.hpp
 */

#ifndef DDS_CORE_STATUS_STATUS_MASK_HPP
#define DDS_CORE_STATUS_STATUS_MASK_HPP

#include <bitset>

namespace dds {
namespace core {
namespace status {
/* 32 bits for StatusMask */
using MaskType = std::bitset<static_cast<size_t>(32)>;

/**
 * @defgroup STATUS_MASK STATUS_MASK
 * @brief A std::bitset (list) of statuses.
 * @par Description
 * 1. refer to the DDS specification for more details (search statusmask)
 * @ingroup STATUS_MASK
 */
class StatusMask : public MaskType {
public:
    using MaskType::MaskType;

    /**
     * @brief All the bits are set.
     */
    static StatusMask All() noexcept
    {
        return StatusMask{0x3007FE7U};
    }

    /**
     * @brief No bits are set.
     */
    static StatusMask None() noexcept
    {
        return StatusMask(0U);
    }

    /**
     * @brief One or more new data samples have been received.
     */
    static StatusMask DataAvailable() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 10U);
    }

    /**
     * @brief The dds::pub::DataWriter has found dds::sub::DataReader that
     * matches the dds::topic::Topic and has compatible QoS.
     */
    static StatusMask PublicationMatched() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 13U);
    }

    /**
     * @brief The dds::sub::DataReader has found dds::pub::DataWriter that
     * matches the dds::topic::Topic and has compatible QoS.
     */
    static StatusMask SubscriptionMatched() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 14U);
    }

    /**
     * @ingroup StatusMask
     * @brief The mask of SampleLost status
     * @param[in] None
     * @return dds::core::status::StatusMask
     * @req {AR20220610482076}
     */
    static StatusMask SampleLost() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 7U);
    }

    /**
     * @ingroup STATUS_MASK
     * @brief One data sample is available for process.
     * @param NONE
     * @return void
     */
    static StatusMask DataProcess() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 24U);
    }

    /**
     * @ingroup StatusMask
     * @brief The mask of SampleTimeOut status
     * @param[in] None
     * @return dds::core::status::StatusMask
     */
    static StatusMask SampleTimeOut() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 25U);
    }

    /**
     * @brief The deadline that the DataWriter has committed through its
     * QosPolicy DEADLINE was not respected for a specific instance
     */
    static StatusMask OfferedDeadlineMissed() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 1U);
    }

    /**
     * @brief The deadline that the DataReader was expecting through its
     * QosPolicy DEADLINE was not respected for a specific instance
     */
    static StatusMask RequestedDealineMissed() noexcept
    {
        return StatusMask(static_cast<uint64_t>(0x0001U) << 2U);
    }
    ~StatusMask() = default;
};
}
}
}

#endif /* DDS_CORE_STATUS_STATUS_MASK_HPP */

