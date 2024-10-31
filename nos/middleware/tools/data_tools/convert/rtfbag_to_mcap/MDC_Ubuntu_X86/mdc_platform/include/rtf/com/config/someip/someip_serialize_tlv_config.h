/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: This provide the config to support rtfcom someip serialze tlv config
 * Create: 2021-05-05
 */

#ifndef RTF_COM_CONFIG_SOMEIP_SERIALIZE_TLV_CONFIG_H
#define RTF_COM_CONFIG_SOMEIP_SERIALIZE_TLV_CONFIG_H

#include "rtf/com/types/ros_types.h"
#include "rtf/com/config/someip/someip_serialize_base_config.h"
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
class SOMEIPSerializeTlvConfig : public SOMEIPSerializeBaseConfig {
public:
    /**
     * @brief SOMEIPSerializeTlvConfig constructor
     */
    explicit SOMEIPSerializeTlvConfig()
        : SOMEIPSerializeBaseConfig(SomeipSerializeConfigFlag::CONFIG_SOMEIP_TLV),
          wireType_(rtf::com::someip::serialize::WireType::STATIC),
          staticLengthField_(rtf::com::someip::serialize::StaticLengthField::FOUR_BYTES) {}

    /**
     * @brief SOMEIPSerializeTlvConfig descontructor
     */
    virtual ~SOMEIPSerializeTlvConfig(void) = default;

    SOMEIPSerializeTlvConfig(SOMEIPSerializeTlvConfig const&) = default;

    SOMEIPSerializeTlvConfig& operator=(SOMEIPSerializeTlvConfig const&) = default;
    /**
     * @brief Set WireType
     * @param[in] wireType tlv serialize wireType config by static or dynamic
     */
    inline void SetWireType(rtf::com::someip::serialize::WireType const wireType) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&wireType);
#endif
        using namespace rtf::com::utils;
        if (wireType > rtf::com::someip::serialize::WireType::DYNAMIC) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            logger_->Warn() << "Wire type is invalid, set failed";
            return;
        }
        wireType_ = wireType;
    }

    /**
     * @brief Return GetWireType
     * @return WireType config
     */
    inline rtf::com::someip::serialize::WireType GetWireType(void) const noexcept
    {
        return wireType_;
    }

    /**
     * @brief Set length filed use how many bytes
     * @param[in] length filed use how many bytes
     */
    inline void SetStaticLengthFieldSize(rtf::com::someip::serialize::StaticLengthField const length) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&length);
#endif
        using namespace rtf::com::utils;
        if (length > rtf::com::someip::serialize::StaticLengthField::FOUR_BYTES) {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1: Logger records the log */
            logger_->Warn() << "Static length field length is invalid, set failed";
            return;
        }
        staticLengthField_ = length;
    }

    /**
     * @brief Return length filed use how many bytes
     * @return length filed use how many bytes
     */
    inline rtf::com::someip::serialize::StaticLengthField GetStaticLengthFieldSize(void) const noexcept
    {
        return staticLengthField_;
    }
private:
    rtf::com::someip::serialize::WireType wireType_;
    rtf::com::someip::serialize::StaticLengthField staticLengthField_;
};
} // namespace config
} // namespace com
} // namespace rtf
#endif // RTF_COM_CONFIG_SOMEIP_SERIALIZE_TLV_CONFIG_H
