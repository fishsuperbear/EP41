/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021 - 2021. All rights reserved.
 * Description: The impl of MsgTimelineRecorder
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_POLICY_DISCOVERYFILTER_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_POLICY_DISCOVERYFILTER_HPP

#include <RT-DDS/dds/core/Types.hpp>
#include <string>

namespace dds {
namespace core {
namespace policy {
class DiscoveryFilter {
public:
    DiscoveryFilter(void) = default;
    ~DiscoveryFilter(void) = default;

    const uint8_t& GetClassificationId(void) const noexcept
    {
        return classificationId_;
    }

    const std::string& GetClassificationIdFilter(void) const noexcept
    {
        return classificationIdFilter_;
    }

    void SetClassificationId(uint8_t classificationId) noexcept
    {
        classificationId_ = classificationId;
    }

    void SetClassificationIdFilter(std::string classificationIdFilter) noexcept
    {
        classificationIdFilter_ = std::move(classificationIdFilter);
    }
private:
    uint8_t classificationId_{};
    std::string classificationIdFilter_;
};
}
}
}
#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_CORE_POLICY_DISCOVERYFILTER_HPP
