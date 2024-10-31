/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: SampleInfo.hpp
 */

#ifndef DDS_SUB_SAMPLE_INFO_HPP
#define DDS_SUB_SAMPLE_INFO_HPP

#include <cstdint>

#include <RT-DDS/dds/sub/state/DataState.hpp>
#include <RT-DDS/dds/core/InstanceHandle.hpp>
#include <RT-DDS/dds/core/Time.hpp>

namespace dds {
using ZeroCopyHandle = uint64_t;
namespace sub {
/**
 * @brief Information that accompanies each sample received by a DataReader.
 */
class SampleInfo {
public:
    /**
     * @brief Identifies locally the corresponding zero copy instance's handle.
     * @return ZeroCopyHandle
     */
    ZeroCopyHandle GetZeroCopyHandle(void) const noexcept
    {
        return zeroCopyHandle_;
    }

    /**
     * @brief Indicates whether the DataSample contains data or else it is only
     * used to communicate a change in the dds::sub::status::InstanceState of
     * the instance.
     * @details Applications should check the valid() flag before accessing the
     * data.
     * @return bool
     */
    bool Valid(void) const noexcept
    {
        return valid_;
    }

    bool GetValid() const noexcept
    {
        return valid_;
    }

    /**
     * @brief Set valid flag.
     * @param[in] v Value of valid flag.
     */
    void Valid(bool v) noexcept
    {
        valid_ = v;
    }

    void SetValid(bool v) noexcept
    {
        valid_ = v;
    }

    /**
     * @brief Set ZeroCopyHandle.
     * @param[in] v Value of ZeroCopyHandle.
     */
    void SetZeroCopyHandle(ZeroCopyHandle v) noexcept
    {
        zeroCopyHandle_ = v;
    }

    uint64_t GetPlogUid(void) const noexcept
    {
        return plogUid_;
    }

    void SetPlogUid(uint64_t plogUid) noexcept
    {
        plogUid_ = plogUid;
    }

    const state::DataState& GetDataState() const noexcept
    {
        return dataState_;
    }

    void SetDataState(const state::DataState& dataState) noexcept
    {
        dataState_ = dataState;
    }

    const core::InstanceHandle& GetInstanceHandle() const noexcept
    {
        return instanceHandle_;
    }

    void SetInstanceHandle(const core::InstanceHandle& instanceHandle) noexcept
    {
        instanceHandle_ = instanceHandle;
    }

    const core::Time& GetSourceTimestamp() const noexcept
    {
        return sourceTimestamp_;
    }

    void SetSourceTimestamp(const core::Time& sourceTimestamp) noexcept
    {
        sourceTimestamp_ = sourceTimestamp;
    }

    uint32_t GetExpediteDataLen () const noexcept
    {
        return expediteDataLen_;
    }

    void SetExpediteDataLen (uint32_t dataLen) noexcept
    {
        expediteDataLen_ = dataLen;
    }

    const void * GetExpediteDataPtr () const noexcept
    {
        return expediteDataPtr_;
    }

    void SetExpediteDataPtr (void * dataptr) noexcept
    {
        expediteDataPtr_ = dataptr;
    }

private:
    ZeroCopyHandle zeroCopyHandle_{0U};
    uint64_t plogUid_{UINT64_MAX};
    state::DataState dataState_{};
    bool valid_{false};
    core::InstanceHandle instanceHandle_{};
    core::Time sourceTimestamp_{};
    uint32_t expediteDataLen_ {0};
    void * expediteDataPtr_ {nullptr};
};
}
}

#endif /* DDS_SUB_SAMPLE_INFO_HPP */

