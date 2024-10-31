/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Types of someip dirver
 * Create: 2019-07-24
 */

#ifndef VRTF_VCC_DRIVER_SOMEIP_SOMEIPDRIVERTYPES_H
#define VRTF_VCC_DRIVER_SOMEIP_SOMEIPDRIVERTYPES_H

#include <functional>
#include <set>
#include "vrtf/vcc/api/types.h"

namespace vrtf {
namespace driver {
namespace someip {
using EventId = std::uint16_t;
using EventGroupId = std::uint16_t;
using MethodId = std::uint16_t;
constexpr EventId ANY_EVENTID = 0xFFFF;
constexpr EventId UNDEFINED_EVENTID = 0xFFFE;
constexpr EventGroupId ANY_EVENTGROUPID = 0xFFFF;
constexpr MethodId ANY_METHODID = 0xFFFF;
constexpr MethodId UNDEFINED_METHODID = 0xFFFE;
class SomeipServiceDiscoveryInfo : public vrtf::vcc::api::types::ServiceDiscoveryInfo {
public:
    SomeipServiceDiscoveryInfo(){}
    ~SomeipServiceDiscoveryInfo(void) = default;
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::SOMEIPTYPE;
    }
    void SetConfigFile(const std::string& file)
    {
        configFile_ = file;
    }
    std::string GetConfigFile() const
    {
        return configFile_;
    }
private:
    std::string configFile_ = "";
};

class SomeipApplicationName {
public:
    static SomeipApplicationName& GetInstance()
    {
        static SomeipApplicationName instance;
        return instance;
    }

    std::string GetName() const
    {
        return vrtf::vcc::api::types::ApplicationName::GetInstance()->GetName();
    }
    void SetDynamicMode(bool isDynamic)
    {
        isDynamicMode_ = isDynamic;
    }

    bool GetDynamicMode() const
    {
        return isDynamicMode_;
    }

    uint16_t GetClientId() const
    {
        return dynamicClientId_;
    }

    void SetClientId(uint16_t clientId)
    {
        dynamicClientId_ = clientId;
    }
private:
    bool isDynamicMode_ = false;
    uint16_t dynamicClientId_ = 0;
};

class SomeipEventInfo : public vrtf::vcc::api::types::EventInfo {
public:
    SomeipEventInfo() {};
    ~SomeipEventInfo(void) = default;
    void SetEventId(const EventId& id)
    {
        eventId_ = id;
    }
    void SetEventGroup(const std::set<EventGroupId>& id)
    {
        eventGroupId_ = id;
    }
    EventId GetEventId() const
    {
        return eventId_;
    }
    std::set<EventGroupId> GetEventGroup() const
    {
        return eventGroupId_;
    }
    bool IsReliable() const
    {
        return isReliable_;
    }
    void SetReliable(bool isReliable)
    {
        isReliable_ = isReliable;
    }
    void SetPort(uint16_t portNum)
    {
        portNum_ = portNum;
        portSet_ = true;
    }
    uint16_t GetPort() const
    {
        return portNum_;
    }
    void SetInstancePort(const std::map<bool, uint16_t>& portWithInstanceId)
    {
        portWithInstanceId_ = portWithInstanceId;
    }
    std::map<bool, uint16_t> GetInstancePort() const
    {
        return portWithInstanceId_;
    }
    bool CheckPortSetted() const
    {
        return portSet_;
    }
    void SetConfigFile(const std::string& file)
    {
        configFile_ = file;
    }
    std::string GetConfigFile() const
    {
        return configFile_;
    }

    std::string GetSomeipEventInfo() const
    {
        std::stringstream eventInfo;
        eventInfo <<  "serviceId=" <<GetServiceId() << ", instanceId=" << GetInstanceId()
            << ", entityId=" << GetEntityId() << ", eventId=" << GetEventId() << ", eventGroup=";
        bool firstEventGroup = true;
        for (const auto& iter : eventGroupId_) {
            if (firstEventGroup) {
                eventInfo << iter;
                firstEventGroup = false;
            } else {
                eventInfo << "." << iter;
            }
        }
        eventInfo << ", majorVersion=" << static_cast<int>(GetVersion().GetMajorVersion());
        return eventInfo.str();
    }
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::SOMEIPTYPE;
    }
    void SetSerializeConfigOfRawData(std::vector<std::uint8_t> const& rawData) { serializeConfigRawData_ = rawData; }
    std::vector<std::uint8_t> GetSerializeConfigOfRawData() const { return serializeConfigRawData_; }
private:
    EventId eventId_ = UNDEFINED_EVENTID;
    std::set<EventGroupId> eventGroupId_;
    std::string configFile_ = "";
    bool isReliable_ = false;
    uint16_t portNum_ = 65535;
    // portWithPubSub_ may be has server or client port with the same instance id.
    std::map<bool, uint16_t> portWithInstanceId_;
    bool portSet_ = false;
    // serializeConfigRawData_ is use to register Someip/S2S serialize config to maintaind
    std::vector<uint8_t> serializeConfigRawData_;
};

class SomeipMethodInfo : public vrtf::vcc::api::types::MethodInfo {
public:
    SomeipMethodInfo() {};
    ~SomeipMethodInfo(void) = default;
    void SetMethodId(const MethodId& id)
    {
        methodId_ = id;
    }
    MethodId GetMethodId() const
    {
        return methodId_;
    }
    void SetReliable(bool isReliable)
    {
        isReliable_ = isReliable;
    }
    bool IsReliable() const
    {
        return isReliable_;
    }
    void SetPort(uint16_t portNum)
    {
        portNum_ = portNum;
    }
    uint16_t GetPort() const
    {
        return portNum_;
    }
    void SetConfigFile(const std::string& file)
    {
        configFile_ = file;
    }
    std::string GetConfigFile() const
    {
        return configFile_;
    }
    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept override
    {
        return vrtf::vcc::api::types::DriverType::SOMEIPTYPE;
    }

    std::string GetMethodInfo() const noexcept
    {
        std::stringstream methodInfo;
        methodInfo <<  "serviceId=" <<GetServiceId() << ", instanceId=" << GetInstanceId()
            << ", entityId=" << GetEntityId() << ", methodId=" << GetMethodId();
        return methodInfo.str();
    }
private:
    MethodId methodId_ = UNDEFINED_METHODID;
    std::string configFile_ = "";
    bool isReliable_ = false;
    uint16_t portNum_ = 0;
};

// Diagnosis types
using DiagnosisCounterType = uint8_t;
using FaultsReportType = uint8_t;

enum class ResetDiagnosisCounterType : uint16_t {
    INVALID_PROTOCOL_VERSION  = 0xb11e,
    INVALID_INTERFACE_VERSION = 0xb11d,
    INVALID_SERVICE_ID        = 0xb11c,
    INVALID_METHOD_ID         = 0xb11b,
    INVALID_MESSAGES          = 0xb11a,
    INVALID_SD_MESSAGES       = 0xb119,
    INVALID_SD_SUBSCRIBE      = 0xb118,
    SERVICE_INDENTIFICATION   = 0x0f05,
    ALL                       = 0xffff
};

enum class FaultsDiagnosisCallbackType : uint8_t {
    COUNTER_CALLBACK = 0x01,
    FAULTS_CALLBACK,
    ALL
};

using DiagnosisCounter = uint16_t;
using CounterCallback  = std::function<void(const DiagnosisCounterType, const DiagnosisCounter)>;
using FaultsCallback   = std::function<void(const uint16_t, const uint16_t, const FaultsReportType)>;
struct FaultsDiagnosisHandler {
    CounterCallback counterCallback = nullptr;
    FaultsCallback  faultsCallback  = nullptr;
};
}
}
}
#endif
