/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: PublicationMatchedStatus.hpp
 */

#ifndef DDS_CORE_STATUS_PUBLICATION_MATCHED_STATUS_HPP
#define DDS_CORE_STATUS_PUBLICATION_MATCHED_STATUS_HPP

#include <cstdint>

namespace dds {
namespace pub {
class AnyDataWriterImpl;
}
}

namespace dds {
namespace core {
namespace status {
/**
 * @brief Information about the status
 * dds::core::status::StatusMask::publication_matched()
 *
 * A "match" happens when the dds::pub::DataWriter finds a dds::sub::DataReader
 * for the same dds::topic::Topic and common partition with a requested QoS that
 * is compatible with that offered by the dds::pub::DataWriter.
 *
 * This status is also changed (and the listener, if any, called) when a match
 * is ended. A local dds::pub::DataWriter will become "unmatched" from a remote
 * dds::sub::DataReader when that dds::sub::DataReader goes away for any reason.
 */
class PublicationMatchedStatus {
public:
    /**
     * @brief The total cumulative number of times the concerned
     * dds::pub::DataWriter discovered a "match" with a dds::sub::DataReader.
     *
     * This number increases whenever a new match is discovered. It does not
     * change when an existing match goes away.
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
     * @brief The current number of readers with which the dds::pub::DataWriter
     * is matched.
     *
     * This number increases when a new match is discovered and decreases when
     * an existing match goes away.
     */
    int32_t CurrentCount() const noexcept
    {
        return currentCount_;
    }

    /**
     * @brief The change in current_count since the last time the listener was
     * called or the status was read.
     */
    int32_t CurrentCountChange() const noexcept
    {
        return currentCountChange_;
    }

    /**
     * @brierf A handle to the last dds::sub::DataReader that caused the the
     * dds::pub::DataWriter's status to change.
     */
    int64_t LastSubscriptionHandle() const noexcept
    {
        return lastSubscriptionHandle_;
    }

private:
    int32_t totalCount_{0};
    int32_t totalCountChange_{0};
    int32_t currentCount_{0};
    int32_t currentCountChange_{0};
    int64_t lastSubscriptionHandle_{0};

    friend class dds::pub::AnyDataWriterImpl;
};
}
}
}

#endif /* DDS_CORE_STATUS_PUBLICATION_MATCHED_STATUS_HPP */

