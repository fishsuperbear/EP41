/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: OfferedDeadlineMissedStatus.hpp
 */

#ifndef DDS_CORE_OFFERED_DEADLINE_MISSED_STATUS_HPP
#define DDS_CORE_OFFERED_DEADLINE_MISSED_STATUS_HPP

#include <cstdint>
#include <RT-DDS/dds/core/InstanceHandle.hpp>

namespace dds {
namespace core {
namespace status {
/**
 * @brief Information about the status
 * dds::core::status::StatusMask::OfferedDeadlineMissedStatus()
 *
 * An "OfferedDeadlineMissed" happens when the dds::pub::DataWriter finds the
 * deadline that the DataWriter has committed through its QosPolicy DEADLINE
 * was not respected for a specioific instance
 */
class OfferedDeadlineMissedStatus {
public:
    /**
     * @brief
     * Total cumulative number of offered deadline periods elapsed during
     * which a DataWriter failed to provide data. Missed deadlines
     * accumulate; that is, each deadline period the total_count will be
     * incremented by one.
     */
    int32_t TotalCount() const noexcept
    {
        return totalCount_;
    }

    /**
    * @brief The incremental changes in total_count since the last time the
     * listener was called or the status was read.
    */
    int32_t TotalCountChange() const noexcept
    {
        return totalCountChange_;
    }

    /**
     * @brierf A handle to the last dds::sub::DataWriter for which an offered
     * deadline was missed
     */
    InstanceHandle LastInstanceHandle() const noexcept
    {
        return lastInstanceHandle_;
    }

private:
    int32_t totalCount_{0};
    int32_t totalCountChange_{0};
    InstanceHandle lastInstanceHandle_{InstanceHandle()};
    friend class dds::pub::AnyDataWriterImpl;
};
}
}
}
#endif
