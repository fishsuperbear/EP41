/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: Define entity(event) index to find corresponding config
 * Create: 2022-11-22
 */
#ifndef RTF_CM_ENTITY_INDEX_INFO_H
#define RTF_CM_ENTITY_INDEX_INFO_H
#include <memory>
#include "ara/core/string.h"
#include "ara/core/result.h"
namespace rtf {
namespace cm {
namespace config {
class CommonEventIndexInfo {
public:
    CommonEventIndexInfo(const std::string &serviceName,
                         const std::uint16_t instanceId,
                         const ara::core::String &eventName,
                         const ara::core::String &vlan) noexcept;
    std::string GetServiceName() const noexcept { return serviceName_; }
    std::uint16_t GetInstanceId() const noexcept { return instanceId_; }
    ara::core::String GetEventName() const noexcept { return eventName_; }
    ara::core::String GetVlan() const noexcept { return vlan_; }
private:
    std::string serviceName_;
    std::uint16_t instanceId_;
    ara::core::String eventName_;
    ara::core::String vlan_;
};

class DDSEventIndexInfo final : public CommonEventIndexInfo {
public:
    DDSEventIndexInfo(const std::string &serviceName, const std::uint16_t instanceId,
                      const ara::core::String &eventName,
                      const ara::core::String &vlan, const std::int16_t domainId) noexcept;
    std::int16_t GetDomainId() const noexcept { return domainId_; };
private:
    std::int16_t domainId_;
};

class SOMEIPEventIndexInfo final : public CommonEventIndexInfo {
public:
    SOMEIPEventIndexInfo(const std::string &serviceName, const std::uint16_t instanceId,
                         const ara::core::String &eventName, const ara::core::String &vlan) noexcept;
};
}  // namespace config
}  // namespace cm
}  // namespace rtf
#endif
