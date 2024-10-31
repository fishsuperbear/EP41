/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: This provide the config to support rtfcom someip serialze config
 * Create: 2021-05-05
 */

#ifndef RTF_COM_CONFIG_SOMEIP_SERIALIZE_BASE_CONFIG_H
#define RTF_COM_CONFIG_SOMEIP_SERIALIZE_BASE_CONFIG_H

#include <unordered_map>

#include "rtf/com/types/ros_types.h"
#include "rtf/com/config/interface/config_info_interface.h"
#include "rtf/com/utils/logger.h"
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
#ifndef COVERITY_TAINT_SET_DEFINITION
#define COVERITY_TAINT_SET_DEFINITION
/**
 * @brief Function for Stain Modeling
 * @details The function is used only when the compilation macro AOS_TAINT is enabled.
 */
static void Coverity_Tainted_Set(void *buf){}
#endif
#endif
namespace rtf {
namespace com {
namespace config {
enum class SomeipSerializeConfigFlag : uint8_t {
    CONFIG_SOMEIP_TLV = 0x00U,
    CONFIG_SOMEIP_RAWDATA = 0x01U,
    CONFIG_SOMEIP_UNDEFINE = 0xFFU,
};
class SOMEIPSerializeBaseConfig {
public:
    SOMEIPSerializeBaseConfig() : logger_(rtf::com::utils::Logger::GetInstance()),isImplementsLegacyStringSerialization_(false),byteOrder_(someip::serialize::ByteOrder::BIGENDIAN) {}

    virtual ~SOMEIPSerializeBaseConfig(void) = default;
    /**
     * @brief Set impleLegacyStringSerialization
     * @param[in] isLegacy impleLegacyStringSerialization serialize config
     */
    void SetImplementsLegacyStringSerialization(const bool isLegacy) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&isLegacy);
#endif
        isImplementsLegacyStringSerialization_ = isLegacy;
    }

    /**
     * @brief Return impleLegacyStringSerialization
     * @return impleLegacyStringSerialization serialize config
     */
    bool GetImplementsLegacyStringSerialization(void) const noexcept
    {
        return isImplementsLegacyStringSerialization_;
    }

    /**
     * @brief Set serialize byte order
     * @param[in] byteOrder serialize byte order config
     */
    inline void SetByteOrder(rtf::com::someip::serialize::ByteOrder const byteOrder) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&byteOrder);
#endif
        using namespace rtf::com::utils;
        if (byteOrder > rtf::com::someip::serialize::ByteOrder::LITTLEENDIAN) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            logger_->Warn() << "Byte order is invalid, set byte order failed";
            return;
        }
        byteOrder_ = byteOrder;
    }

    /**
     * @brief Return byte order config
     * @return byte order config
     */
    rtf::com::someip::serialize::ByteOrder GetByteOrder() const noexcept
    {
        return byteOrder_;
    }

    /**
     * @brief Return serialize config is tlv config
     * @return serialize config is tlv config
     */
    bool GetSerializeTypeIsTLV() const noexcept
    {
        return serializeConfigFlag_ == SomeipSerializeConfigFlag::CONFIG_SOMEIP_TLV;
    }
    bool GetSerializeTypeIsRawData() const noexcept
    {
        return serializeConfigFlag_ == SomeipSerializeConfigFlag::CONFIG_SOMEIP_RAWDATA;
    }
protected:
    explicit SOMEIPSerializeBaseConfig(const SomeipSerializeConfigFlag serializeConfigFlag)
        : logger_(rtf::com::utils::Logger::GetInstance()),isImplementsLegacyStringSerialization_(false), serializeConfigFlag_(serializeConfigFlag),
          byteOrder_(someip::serialize::ByteOrder::BIGENDIAN){}
    SOMEIPSerializeBaseConfig(const SOMEIPSerializeBaseConfig& other) = default;
    SOMEIPSerializeBaseConfig& operator=(const SOMEIPSerializeBaseConfig& other) = default;
    std::shared_ptr<rtf::com::utils::Logger> logger_;
private:
    bool isImplementsLegacyStringSerialization_;
    SomeipSerializeConfigFlag serializeConfigFlag_ = SomeipSerializeConfigFlag::CONFIG_SOMEIP_UNDEFINE;
    someip::serialize::ByteOrder byteOrder_;
};
} // namespace config
} // namespace com
} // namespace rtf
#endif // RTF_COM_CONFIG_SOMEIP_SERIALIZE_BASE_CONFIG_H
