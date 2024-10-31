/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This config is support rtftools register serialize config to maintaind
 * Create: 2022-06-08
 */

#ifndef RTF_COM_CONFIG_SOMEIP_SERIALIZE_RAWDATA_CONFIG_H
#define RTF_COM_CONFIG_SOMEIP_SERIALIZE_RAWDATA_CONFIG_H
#include <vector>
#include "rtf/com/config/someip/someip_serialize_base_config.h"
namespace rtf {
namespace com {
namespace config {
class SOMEIPSerializeRawDataConfig : public SOMEIPSerializeBaseConfig {
public:
    SOMEIPSerializeRawDataConfig() : SOMEIPSerializeBaseConfig(SomeipSerializeConfigFlag::CONFIG_SOMEIP_RAWDATA) { }
    virtual ~SOMEIPSerializeRawDataConfig() = default;

    SOMEIPSerializeRawDataConfig(SOMEIPSerializeRawDataConfig const&) = default;
    SOMEIPSerializeRawDataConfig& operator=(SOMEIPSerializeRawDataConfig const&) = default;
    void SetSerializeConfigOfRawData(std::vector<std::uint8_t> const& rawData) { serializeConfigRawData_ = rawData; }
    std::vector<std::uint8_t> GetSerializeConfigOfRawData() const { return serializeConfigRawData_; }
private:
    std::vector<std::uint8_t> serializeConfigRawData_;
};
} // namespace config
} // namespace com
} // namespace rtf
#endif