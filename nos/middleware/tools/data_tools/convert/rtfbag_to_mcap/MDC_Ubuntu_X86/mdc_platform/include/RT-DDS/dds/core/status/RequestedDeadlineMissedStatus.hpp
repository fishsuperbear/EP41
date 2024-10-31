/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: RequestedDeadlineMissedStatus.hpp
 */
#ifndef DDS_CORE_REQUESTED_DEADLINE_MISSED_STATUS_H
#define DDS_CORE_REQUESTED_DEADLINE_MISSED_STATUS_H

#include <cstdint>
#include "RT-DDS/dds/core/InstanceHandle.hpp"

namespace dds {
namespace core {
namespace status {
/**
 * @brief Information about the status
 * dds::core::status::StatusMask::RequestedDealineMissed()
 *
 * An "RequestedDeadlineMissed" happens when the dds::pub::DataReader finds
 * The deadline that the DataReader was expecting through its QosPolicy
 * DEADLINE was not respected for a specific instance
 */
class RequestedDeadlineMissedStatus {
public:
    /**
     * @brief  Total cumulative number of missed deadlines detected for any instance
     * read by the DataReader. Missed deadlines accumulate; that is, each
     * deadline period the total_count will be incremented by one for each
     * instance for which data was not received.
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
     * @brierf A handle to the last dds::sub::DataReader that caused the the
     * dds::pub::DataReader's status to change.
     */
    InstanceHandle LastInstanceHandle() const noexcept
    {
        return lastInstanceHandle_;
    }

private:
    int32_t totalCount_{0};
    int32_t totalCountChange_{0};
    InstanceHandle lastInstanceHandle_{InstanceHandle()};

    friend class dds::sub::DataReaderImpl;
};
}
}
}
#endif
