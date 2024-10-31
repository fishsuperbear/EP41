/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Types of someip dirver
 * Create: 2019-07-24
 */

#ifndef VRTF_VCC_DRIVER_PROLOC_PROLOCDRIVERTYPES_H
#define VRTF_VCC_DRIVER_PROLOC_PROLOCDRIVERTYPES_H

#include <functional>
#include <set>
#include "vrtf/vcc/api/types.h"

namespace vrtf {
namespace driver {
namespace proloc {
using ServiceId = vrtf::vcc::api::types::ServiceId;
using InstanceId = vrtf::vcc::api::types::InstanceId;
using EntityId = vrtf::vcc::api::types::EntityId;
using ClientUid = std::size_t;
class ProlocEntityIndex final {
public:
    ProlocEntityIndex(
        ServiceId const serviceId, InstanceId const &instanceId, EntityId const entityId)
        : serviceId_(serviceId), instanceId_(instanceId), entityId_(entityId) {}
    ~ProlocEntityIndex() = default;
    ProlocEntityIndex(const ProlocEntityIndex& other) = default;
    ProlocEntityIndex& operator=(ProlocEntityIndex const &prolocEntityIndex) = default;
    EntityId GetEntityId() const noexcept
    {
        return entityId_;
    }
    InstanceId GetInstanceId() const noexcept
    {
        return instanceId_;
    }
    ServiceId GetServiceId() const noexcept
    {
        return serviceId_;
    }

    std::string GetProlocInfo() const noexcept
    {
        std::stringstream prolocInfo;
        prolocInfo << "serviceId=" << serviceId_ << ", instanceId=" << instanceId_ << ", entityId=" << entityId_;
        return prolocInfo.str();
    }
    bool operator<(const ProlocEntityIndex &other) const noexcept
    {
        if (serviceId_ != other.serviceId_) {
            return (serviceId_ < other.serviceId_);
        } else if (instanceId_ != other.instanceId_) {
            return (instanceId_ < other.instanceId_);
        } else {
            return (entityId_ < other.entityId_);
        }
    }

    bool operator==(const ProlocEntityIndex &other) const noexcept
    {
        if ((instanceId_ == other.instanceId_) && (entityId_ == other.entityId_) &&
            (serviceId_ == other.serviceId_)) {
            return true;
        }
        return false;
    }
private:
    ServiceId serviceId_;
    InstanceId instanceId_;
    EntityId entityId_;
};


class ProlocServiceDiscoveryInfo : public vrtf::vcc::api::types::ServiceDiscoveryInfo {
public:
    ProlocServiceDiscoveryInfo() {}
    ~ProlocServiceDiscoveryInfo(void) override = default;
    ProlocServiceDiscoveryInfo(const ProlocServiceDiscoveryInfo& other) = default;
    ProlocServiceDiscoveryInfo& operator=(ProlocServiceDiscoveryInfo const &prolocServiceDiscoveryInfo) = default;
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::PROLOCTYPE;
    }
};

class ProlocEventInfo : public vrtf::vcc::api::types::EventInfo {
public:
    ProlocEventInfo() = default;
    ~ProlocEventInfo(void) override = default;
    ProlocEventInfo(const ProlocEventInfo& other) = default;
    ProlocEventInfo& operator=(ProlocEventInfo const &prolocEventInfo) = default;
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::PROLOCTYPE;
    }
};

class ProlocMethodInfo : public vrtf::vcc::api::types::MethodInfo {
public:
    ProlocMethodInfo() = default;
    ~ProlocMethodInfo(void) override = default;
    ProlocMethodInfo(const ProlocMethodInfo& other) = default;
    ProlocMethodInfo& operator=(ProlocMethodInfo const &prolocMethodInfo) = default;
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::PROLOCTYPE;
    }
};

class ProlocMethodMsg : public vrtf::vcc::api::types::MethodMsg {
public:
explicit  ProlocMethodMsg(ClientUid uid) : vrtf::vcc::api::types::MethodMsg(), clientUid_(uid) {}
~ProlocMethodMsg() override = default;
ProlocMethodMsg(const ProlocMethodMsg& other) = default;
ProlocMethodMsg& operator=(ProlocMethodMsg const &prolocMethodMsg) = default;
ClientUid GetClientUid() const noexcept { return clientUid_; }
private:
ClientUid clientUid_;
};
}
}
}
#endif
