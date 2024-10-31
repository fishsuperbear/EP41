/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataReaderProtocol.hpp
 */

#ifndef DDS_CORE_POLICY_DATA_READER_PROTOCOL_HPP
#define DDS_CORE_POLICY_DATA_READER_PROTOCOL_HPP

#include <RT-DDS/dds/core/policy/RtpsReliableReaderProtocol.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures DataReader-specific aspects of the RTPS protocol.
 */
class DataReaderProtocol {
public:
    /**
     * @brief RTPS protocol-related configuration settings for the RTPS reliable
     * reader associated to a dds::sub::DataReader. This parameter only has
     * effect if the reader is configured with dds::core::policy::ReliabilityKind.
     * @param p The reliable protocol settings.
     */
    void RtpsReliableReader(RtpsReliableReaderProtocol p) noexcept
    {
        rtpsReliableReader_ = p;
    }

    /**
     * @brief Gets the reliable protocol settings by const reference.
     * @return The reliable protocol settings.
     * @see void RtpsReliableReader(RtpsReliableReaderProtocol p).
     */
    const RtpsReliableReaderProtocol &RtpsReliableReader() const noexcept
    {
        return rtpsReliableReader_;
    }

private:
    RtpsReliableReaderProtocol rtpsReliableReader_{};
};
}
}
}

#endif /* DDS_CORE_POLICY_DATA_READER_PROTOCOL_HPP */

