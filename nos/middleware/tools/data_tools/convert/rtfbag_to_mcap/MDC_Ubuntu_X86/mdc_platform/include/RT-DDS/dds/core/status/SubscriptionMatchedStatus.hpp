/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SubscriptionMatchedStatus.hpp
 */

#ifndef DDS_CORE_SUBSCRIPTION_MATCHED_STATUS_HPP
#define DDS_CORE_SUBSCRIPTION_MATCHED_STATUS_HPP

#include <cstdint>

namespace dds {
namespace sub {
class DataReaderImpl;
}
}

namespace dds {
namespace core {
namespace status {
/**
 * @brief Information about the status
 * dds::core::status::StatusMask::subscription_matched()
 *
 * A "match" happens when the dds::sub::DataReader finds a dds::pub::DataWriter
 * for the same dds::topic::Topic with an offered QoS that is compatible with
 * that requested by the dds::sub::DataReader.
 *
 * This status is also changed (and the listener, if any, called) when a match
 * is ended. A local dds::sub::DataReader will become "unmatched" from a remote
 * dds::pub::DataWriter when that dds::pub::DataWriter goes away for any reason.
 */
class SubscriptionMatchedStatus {
public:
    /**
     * @brief The total cumulative number of times the concerned
     * dds::sub::DataReader discovered a "match" with a dds::pub::DataWriter.
     *
     * This number increases whenever a new match is discovered. It does not
     * change when an existing match goes away.
     */
    int32_t TotalCount() const noexcept
    {
        return totalCount_;
    }

    /**
    * @brief The change in total_count since the last time the listener was
    * called or the status was read.
    */
    int32_t TotalCountChange() const noexcept
    {
        return totalCountChange_;
    }

    /**
     * @brief The current number of writers with which the dds::sub::DataReader
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
     * @brierf A handle to the last dds::pub::DataWriter that caused the status
     * to change.
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

    friend class dds::sub::DataReaderImpl;
};
}
}
}

#endif /* DDS_CORE_SUBSCRIPTION_MATCHED_STATUS_HPP */

