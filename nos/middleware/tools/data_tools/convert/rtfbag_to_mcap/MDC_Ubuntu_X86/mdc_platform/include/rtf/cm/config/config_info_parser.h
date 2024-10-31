/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This file just for rtf_tool get bag config
 * Create: 2022-11-22
 */
#ifndef RTF_CM_CONFIG_CONFIG_INFO_PARSER_H
#define RTF_CM_CONFIG_CONFIG_INFO_PARSER_H
#include <memory>
#include "rtf/cm/config/entity_index_info.h"
#include "vrtf/driver/dds/dds_driver_types.h"
#include "vrtf/driver/someip/someip_driver_types.h"
#include "ara/core/string.h"
#include "ara/core/result.h"
namespace rtf {
namespace cm {
namespace config {
class ConfigInfoParser {
public:
    ConfigInfoParser() = default;
    virtual ~ConfigInfoParser() = default;
    ConfigInfoParser(const ConfigInfoParser&) = delete;
    ConfigInfoParser(ConfigInfoParser &&) = delete;
    ConfigInfoParser &operator=(ConfigInfoParser &&) & = delete;
    ConfigInfoParser& operator=(ConfigInfoParser const &) & = delete;
    virtual ara::core::Result<vrtf::driver::dds::DDSEventInfo> ParserDDSEvent(
        const DDSEventIndexInfo &indexInfo) const = 0;
    virtual ara::core::Result<vrtf::driver::someip::SomeipEventInfo> ParserSOMEIPEvent(
        const SOMEIPEventIndexInfo &indexInfo) const = 0;
};

//  Prohibited use !!!
class ConfigParserFactory {
public:
    static ara::core::Result<std::unique_ptr<ConfigInfoParser>> CreateConfigParser(const std::string &path);
};
}  // namespace config
}  // namespace cm
}  // namespace rtf
#endif
