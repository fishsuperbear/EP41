/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataWriterProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_DATA_WRITER_PROTOCOL_HPP
#define DDS_CORE_POLICY_DATA_WRITER_PROTOCOL_HPP

#include <RT-DDS/dds/core/policy/RtpsReliableWriterProtocol.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures aspects of an the RTPS protocol related to a
 * dds::pub::DataWriter
 */
class DataWriterProtocol {
public:
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * writer associated to a dds::pub::DataWriter. This parameter only has
     * effect if both the writer and the matching reader are configured with
     * dds::core::policy::ReliabilityKind::RELIABLE
     */
    void RtpsReliableWriter(RtpsReliableWriterProtocol p) noexcept
    {
        rtpsReliableWriter_ = p;
    }

    /**
     * Gets the reliable settings by const-reference
     * @see void RtpsReliableWriter(RtpsReliableWriterProtocol p).
     */
    const RtpsReliableWriterProtocol &RtpsReliableWriter() const noexcept
    {
        return rtpsReliableWriter_;
    }

private:
    RtpsReliableWriterProtocol rtpsReliableWriter_{};
};
}
}
}

#endif /* DDS_CORE_POLICY_DATA_WRITER_PROTOCOL_HPP */

