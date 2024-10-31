/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_SUB_TAKEPARAMS_HPP
#define SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_SUB_TAKEPARAMS_HPP

#include <cstdint>

#include <RT-DDS/dds/core/InstanceHandle.hpp>

namespace dds {
namespace sub {
/**
 * @brief A class to help determine the params of one take operation
 */
class TakeParams {
public:

    /**
     * @brief the max samples would take in one take operations
     */
    TakeParams& SetMaxSample(int32_t maxSample) noexcept
    {
        maxSample_ = maxSample;
        return *this;
    }

    /**
     * @brief the target instance user would be interested in one take operations
     */
    TakeParams& SetInstanceHandle(const core::InstanceHandle& instanceHandle) noexcept
    {
        instanceHandle_ = instanceHandle;
        return *this;
    }

    int32_t GetMaxSample() const noexcept
    {
        return maxSample_;
    }

    const core::InstanceHandle& GetInstanceHandle() const noexcept
    {
        return instanceHandle_;
    }

private:
    int32_t maxSample_{1};
    dds::core::InstanceHandle instanceHandle_{core::InstanceHandle::Nil()};
};
}
}


#endif // SRC_DCPS_API_ISOCPP_INCLUDE_RT_DDS_DDS_SUB_TAKEPARAMS_HPP
