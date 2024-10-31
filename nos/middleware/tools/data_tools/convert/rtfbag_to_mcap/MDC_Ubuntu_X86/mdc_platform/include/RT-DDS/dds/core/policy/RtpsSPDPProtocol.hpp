/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: RtpsSPDPProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_RTPS_SPDP_PROTOCOL
#define DDS_CORE_POLICY_RTPS_SPDP_PROTOCOL

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
* @brief Configures aspects of the RTPS protocol related to SPDP announcements.
*
* It is used to configure SPDP announcements after a participant discovers another for the first time.
*/
class RtpsSPDPProtocol {
public:
    /**
     * @brief  The unicast SPDP announcement times.
     *
     * This value determines how many times a participant sends SPDP announcements
     * to a newly-discovered counterpart via unicast. Note that the announcements
     * will automatically stop when it receives the SEDP messages from the opposite.
     */
    void Announcements(uint8_t times) noexcept
    {
        announcements_ = times;
        isAssignedByUser_ = true;
    }

    /**
     * @brief The minimum interval between unicast SPDP announcements.
     *
     * This value provides the lower bound of the random time interval between
     * consecutive unicast SPDP announcements.
     */
    void MinAnnouncementInterval(uint32_t min) noexcept
    {
        minAnnouncementInterval_ = min;
    }

    /**
    * @brief The maximum interval between unicast SPDP announcements.
    *
    * This value provides the upper bound of the random time interval between
    * consecutive unicast SPDP announcements.
    */
    void MaxAnnouncementInterval(uint32_t max) noexcept
    {
        maxAnnouncementInterval_ = max;
    }
    /**
     * @brief Getter (see setter with the same name)
     */
    const uint8_t &Announcements() const noexcept
    {
        return announcements_;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const uint32_t &MinAnnouncementInterval() const noexcept
    {
        return minAnnouncementInterval_;
    }

    /**
     * @brief Getter (see setter with the same name)
     */
    const uint32_t &MaxAnnouncementInterval() const noexcept
    {
        return maxAnnouncementInterval_;
    }

    /**
     * @brief states whether the announcements_ field is assigned by user.
     */
    const bool &IsAssignedByUser() const noexcept
    {
        return isAssignedByUser_;
    }

private:
    uint8_t announcements_{0U};  /* stands for unlimited resending. */
    uint32_t minAnnouncementInterval_{500U};
    uint32_t maxAnnouncementInterval_{1000U};
    bool isAssignedByUser_{false};
};
}
}
}

#endif /* DDS_CORE_POLICY_RTPS_SPDP_PROTOCOL */

