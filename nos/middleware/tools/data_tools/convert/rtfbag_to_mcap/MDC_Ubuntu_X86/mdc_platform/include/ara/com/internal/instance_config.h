/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: This file provides an interface related to communication dynamic config.
 * Create: 2019-07-01
 */
#ifndef ARA_COM_INSTANCE_CONFIG_H
#define ARA_COM_INSTANCE_CONFIG_H
#include <memory>
#include "ara/com/types.h"
#include "vrtf/driver/dds/dds_driver_types.h"

namespace ara {
namespace com {
namespace dds {
using DomainId = vrtf::driver::dds::DomainId;
using QosPolicy = vrtf::driver::dds::QosPolicy;
}  // namespace dds
class InstanceConfig {
public:
    InstanceConfig() = default;
    virtual ~InstanceConfig() = default;
    static std::shared_ptr<InstanceConfig>
    Create(const std::string &serviceName, const std::shared_ptr<vrtf::vcc::api::types::ServiceDiscoveryInfo> &sdInfo,
           const std::vector<std::shared_ptr<vrtf::vcc::api::types::EventInfo> > &eventInfo,
           const std::vector<std::shared_ptr<vrtf::vcc::api::types::MethodInfo> > &methodInfo,
           const std::vector<std::shared_ptr<vrtf::vcc::api::types::MethodInfo> > &fieldInfo);
    virtual bool Register() = 0;
private:
    InstanceConfig(const InstanceConfig& other) = default;
    InstanceConfig(InstanceConfig&& other) = default;
    InstanceConfig& operator=(const InstanceConfig& other) = default;
    InstanceConfig& operator=(InstanceConfig&& other) = default;
};
}  // namespace com
}  // namespace ara
#endif
