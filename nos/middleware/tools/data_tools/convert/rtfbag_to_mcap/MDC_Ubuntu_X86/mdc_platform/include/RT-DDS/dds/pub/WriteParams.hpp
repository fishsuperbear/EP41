/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: WriteParams.hpp
 */

#ifndef DDS_CORE_WRITE_PARAMS_HPP
#define DDS_CORE_WRITE_PARAMS_HPP

#include <cstdint>
#include <RT-DDS/dds/core/Guid.hpp>
#include <RT-DDS/dds/core/InstanceHandle.hpp>
#include <RT-DDS/dds/core/ReturnCode.hpp>
#include <RT-DDS/dds/core/Time.hpp>
#include <plog/PLogDefsAndLimits.hpp>

namespace dds {
using ZeroCopyHandle = uint64_t;
namespace pub {
struct AllocateResult {
    dds::core::ReturnCode res{};
    ZeroCopyHandle allocatedHandle{};
};

class WriteParams {
public:
    void SetZeroCopyHandle(ZeroCopyHandle handle) noexcept
    {
        handle_ = handle;
    }

    ZeroCopyHandle GetZeroCopyHandle(void) const noexcept
    {
        return handle_;
    }

    void RelatedSourceGuid(dds::core::Guid relatedSourceGuid) noexcept
    {
        relatedSourceGuid_ = relatedSourceGuid;
    }

    void SetRelatedSourceGuid(dds::core::Guid relatedSourceGuid) noexcept
    {
        relatedSourceGuid_ = relatedSourceGuid;
    }

    dds::core::Guid RelatedSourceGuid(void) const noexcept
    {
        return relatedSourceGuid_;
    }

    dds::core::Guid GetRelatedSourceGuid(void) const noexcept
    {
        return relatedSourceGuid_;
    }

    void SetPlogUid(rbs::plog::PlogUid uid) noexcept
    {
        WriteParams::plogUid_ = uid;
    }

    rbs::plog::PlogUid GetPlogUid(void) const noexcept
    {
        return plogUid_;
    }

    const core::InstanceHandle& GetInstanceHandle() const noexcept
    {
        return instanceHandle_;
    }

    void SetInstanceHandle(const core::InstanceHandle& instanceHandle) noexcept
    {
        instanceHandle_ = instanceHandle;
    }

    const core::Time& GetTimestamp() const noexcept
    {
        return timestamp_;
    }

    void SetTimestamp(const core::Time& timestamp) noexcept
    {
        timestamp_ = timestamp;
    }

private:
    ZeroCopyHandle handle_{0U};
    dds::core::Guid relatedSourceGuid_{};
    rbs::plog::PlogUid plogUid_{rbs::plog::PLOG_UID_MAX};
    dds::core::InstanceHandle instanceHandle_{};
    dds::core::Time timestamp_{dds::core::Time::Invalid()};
};
}
}

#endif // DDS_CORE_WRITE_PARAMS_HPP

