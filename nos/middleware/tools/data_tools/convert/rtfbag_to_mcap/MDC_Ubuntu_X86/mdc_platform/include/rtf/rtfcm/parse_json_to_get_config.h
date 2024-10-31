/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Parse rtf.json to get config.
 * Create: 2021-06-19
 */
#ifndef PARSE_JSON_TO_GET_CONFIG_H
#define PARSE_JSON_TO_GET_CONFIG_H
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/api/types.h"
#include "vrtf/driver/dds/dds_qos_store.h"
#include "json_parser/rtf_config_parser.h"
namespace rtf {
namespace rtfcm {
namespace rtfmaintaind {
class ParseJsonToGetConfig {
public:
    using RtfConfigParser = ara::godel::common::rtfConfigParser::RtfConfigParser;
    ParseJsonToGetConfig();
    // The below ctor just used by ut
    explicit ParseJsonToGetConfig(const std::string &resouceFile);
    ~ParseJsonToGetConfig() = default;
    static std::shared_ptr<ParseJsonToGetConfig>& GetInstance() noexcept;
    vrtf::vcc::api::types::ResourceAttr GetResourceAttr() const;
    vrtf::vcc::api::types::NetworkIp GetNetworkIp() const;
    vrtf::vcc::api::types::InstanceId GetInstanceId() const;
    vrtf::driver::dds::qos::TransportMode GetTransportMode() const;
private:
    using TransportMode = vrtf::driver::dds::qos::TransportMode;
    void Init(RtfConfigParser const &parser);
    void ParseResourceAttr(RtfConfigParser const &parser);
    void ParseRtfConfigNetwork(const RtfConfigParser &parser);
    vrtf::vcc::api::types::InstanceId ParseToGetInstanceId(const RtfConfigParser &parser) const;
    void SetInstanceId(const RtfConfigParser &parser);
    void ParseTransportMode(RtfConfigParser const &parser);

    vrtf::vcc::api::types::ResourceAttr resourceAttr_;
    vrtf::vcc::api::types::NetworkIp networkIp_;
    vrtf::vcc::api::types::InstanceId instanceId_ {vrtf::vcc::api::types::UNDEFINED_INSTANCEID};
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    TransportMode transportMode_{TransportMode::TRANSPORT_ASYNCHRONOUS_MODE};
};
}
}
}
#endif
