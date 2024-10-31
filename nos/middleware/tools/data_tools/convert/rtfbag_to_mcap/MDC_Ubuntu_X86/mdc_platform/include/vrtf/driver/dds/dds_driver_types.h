/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Types of dds driver
 * Create: 2019-07-24
 */
#ifndef VRTF_VCC_DRIVER_DDS_DDSDRIVERTYPES_H
#define VRTF_VCC_DRIVER_DDS_DDSDRIVERTYPES_H
#include <functional>
#include <string>
#include <memory>
#include "vrtf/vcc/api/types.h"
#include "vrtf/driver/dds/dds_qos_store.h"
namespace vrtf {
namespace driver {
namespace dds {
using DomainId = std::int16_t;
using NetworkIp = std::pair<std::string, bool>;
enum class QosPolicy : uint8_t {
    LARGE,
    DEFAULT,
    SMALL
};
class DDSServiceDiscoveryInfo final : public vcc::api::types::ServiceDiscoveryInfo {
public:
    DDSServiceDiscoveryInfo() = default;
    ~DDSServiceDiscoveryInfo(void) final = default;

    DDSServiceDiscoveryInfo(const DDSServiceDiscoveryInfo& other) = default;
    DDSServiceDiscoveryInfo& operator=(const DDSServiceDiscoveryInfo& other) & = default;
    DDSServiceDiscoveryInfo(DDSServiceDiscoveryInfo&& other) = default;
    DDSServiceDiscoveryInfo& operator=(DDSServiceDiscoveryInfo&& other) & = default;

    DomainId GetDomainId() const noexcept
    {
        return domainId_;
    }
    void SetDomainId(DomainId id) noexcept
    {
        domainId_ = id;
    }
    std::string GetQosPath() const noexcept
    {
        return qosPath_;
    }
    void SetQosPath(const std::string& path) noexcept
    {
        qosPath_ = path;
    }

    void SetParticipantTransportQos(const std::set<vrtf::driver::dds::qos::TransportQos>& transportMode) noexcept
    {
        transportQos_ = transportMode;
    }

    std::set<vrtf::driver::dds::qos::TransportQos> GetParticipantTransportQos() const noexcept
    {
        return transportQos_;
    }

    void SetParticipantQos(const qos::ParticipantQos& participantQos) noexcept
    {
        participantQos_ = participantQos;
    }

    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept final
    {
        return vrtf::vcc::api::types::DriverType::DDSTYPE;
    }

    qos::ParticipantQos GetParticipantQos() const noexcept
    {
        return participantQos_;
    }

private:
    DomainId domainId_ {0U};
    std::string qosPath_ {""};
    std::set<vrtf::driver::dds::qos::TransportQos> transportQos_ = {vrtf::driver::dds::qos::TransportQos::UDP};
    qos::ParticipantQos participantQos_ {qos::ParticipantQos(qos::DiscoveryFilter(0, "UNDEFINED_DISCOVERY_FILTER"))};
};

class DDSEventInfo final: public vcc::api::types::EventInfo {
public:
    DDSEventInfo() = default;
    ~DDSEventInfo(void) final = default;

    DDSEventInfo(const DDSEventInfo& other) = default;
    DDSEventInfo& operator=(const DDSEventInfo& other) & = default;
    DDSEventInfo(DDSEventInfo&& other) = default;
    DDSEventInfo& operator=(DDSEventInfo&& other) & = default;

    qos::DDSEventQosStore ddsQos_;
    inline std::string GetTopicName() const
    {
        return topicName_;
    }
    inline void SetTopicName(const std::string& name)
    {
        topicName_ = name;
    }
    inline DomainId GetDomainId() const noexcept
    {
        return domainId_;
    }
    inline void SetDomainId(DomainId id) noexcept
    {
        domainId_ = id;
    }
    inline std::string GetQosProfile() const
    {
        return qosProfile_;
    }
    inline void SetQosProfile(const std::string& profile)
    {
        qosProfile_ = profile;
    }

    void SetAttribute(const std::map<std::string, std::string>& attributeValueList)
    {
        ddsAttributeList_ = attributeValueList;
    }
    std::map<std::string, std::string> GetAtrribute() const
    {
        return ddsAttributeList_;
    }

    inline void SetScheduleMode(qos::ScheduleMode scheduleMode)
    {
        scheduleMode_ = scheduleMode;
    }

    inline qos::ScheduleMode GetScheduleMode() const
    {
        return scheduleMode_;
    }
    void SetParticipantQos(const qos::ParticipantQos& participantQos) noexcept
    {
        participantQos_ = participantQos;
    }

    qos::ParticipantQos GetParticipantQos() const noexcept
    {
        return participantQos_;
    }

    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept final
    {
        return vrtf::vcc::api::types::DriverType::DDSTYPE;
    }
    void SetQosPolicy(QosPolicy policy)
    {
        static_cast<void>(policy);
        ddsQos_.SetDurabilityQos(qos::DurabilityQos::VOLATILE);
        ddsQos_.SetHistoryQos(qos::HistoryQos::KEEP_ALL);
    }

    void SetMbufByPoolFlag(bool enable) noexcept
    {
        mbufByPoolFlag_ = enable;
    }

    bool GetMbufByPoolFlag() const noexcept
    {
        return mbufByPoolFlag_;
    }

    void SetDirectProcessFlag(bool flag) noexcept
    {
        directProcessFlag_ = flag;
    }

    bool GetDirectProcessFlag() const noexcept
    {
        return directProcessFlag_;
    }

    std::string GetEventInfo() const noexcept
    {
        std::stringstream eventInfo;
        eventInfo << "serviceId=" << GetServiceId() << ", instanceId=" << GetInstanceId() << ", entityId=" <<
            GetEntityId() << ", topicName=" << GetTopicName();
        return eventInfo.str();
    }
private:
    DomainId domainId_ {0U};
    std::string topicName_ {""};
    std::string qosProfile_ {""};
    qos::ScheduleMode scheduleMode_ {qos::ScheduleMode::DETERMINATE};
    std::map<std::string, std::string> ddsAttributeList_ {};
    qos::ParticipantQos participantQos_ {qos::ParticipantQos(qos::DiscoveryFilter(0, "UNDEFINED_DISCOVERY_FILTER"))};
    bool mbufByPoolFlag_ {false};
    bool directProcessFlag_ {false};
};

class DDSMethodInfo final: public vcc::api::types::MethodInfo {
public:
    DDSMethodInfo() = default;
    ~DDSMethodInfo(void) final = default;

    DDSMethodInfo(const DDSMethodInfo& other) = default;
    DDSMethodInfo& operator=(const DDSMethodInfo& other) & = default;
    DDSMethodInfo(DDSMethodInfo&& other) = default;
    DDSMethodInfo& operator=(DDSMethodInfo&& other) & = default;

    qos::DDSMethodQosStore ddsMethodQos_;
    inline std::string GetRequestTopicName() const noexcept
    {
        return requestTopicName_;
    }
    inline void SetRequestTopicName(const std::string& name) noexcept
    {
        requestTopicName_ = name;
    }
    inline std::string GetReplyTopicName() const noexcept
    {
        return replyTopicName_;
    }
    inline void SetReplyTopicName(const std::string& name) noexcept
    {
        replyTopicName_ = name;
    }
    inline std::string GetQosProfile() const noexcept
    {
        return qosProfile_;
    }
    inline void SetQosProfile(const std::string& profile) noexcept
    {
        qosProfile_ = profile;
    }
    DomainId GetDomainId() const noexcept
    {
        return domainId_;
    }
    void SetDomainId(DomainId id) noexcept
    {
        domainId_ = id;
    }
    uint32_t GetMethodNum() const noexcept
    {
        return methodNum_;
    }
    void SetParticipantQos(const qos::ParticipantQos& participantQos) noexcept
    {
        participantQos_ = participantQos;
    }

    qos::ParticipantQos GetParticipantQos() const noexcept
    {
        return participantQos_;
    }

    void SetQosPolicy(QosPolicy policy) noexcept
    {
        // 4 array elements meaning: request writer depth, request reader depth, reply writer depth, reply reader depth
        std::map<QosPolicy, std::array<std::int32_t, 4>> depthMap {
            {QosPolicy::LARGE, {5, 100, 100, 5}},
            {QosPolicy::DEFAULT, {5, 20, 3, 5}},
            {QosPolicy::SMALL, {5, 10, 3, 5}}
        };
        constexpr std::size_t requestWriterDepth {0U};
        constexpr std::size_t requestReaderDepth {1U};
        constexpr std::size_t replyWriterDepth {2U};
        constexpr std::size_t replyReaderDepth {3U};
        ddsMethodQos_.SetRequestWriterHistoryDepth(depthMap[policy][requestWriterDepth]); // request writer depth
        ddsMethodQos_.SetRequestReaderHistoryDepth(depthMap[policy][requestReaderDepth]); // request reader depth
        ddsMethodQos_.SetReplyWriterHistoryDepth(depthMap[policy][replyWriterDepth]); // reply writer depth
        ddsMethodQos_.SetReplyReaderHistoryDepth(depthMap[policy][replyReaderDepth]); // reply reader depth
        ddsMethodQos_.SetRequestHistoryQos(qos::HistoryQos::KEEP_ALL);
        ddsMethodQos_.SetReplyHistoryQos(qos::HistoryQos::KEEP_ALL);
        ddsMethodQos_.SetRequestDurabilityQos(qos::DurabilityQos::VOLATILE);
        ddsMethodQos_.SetReplyDurabilityQos(qos::DurabilityQos::VOLATILE);
    }

    vrtf::vcc::api::types::DriverType GetDriverType() const noexcept final
    {
        return vrtf::vcc::api::types::DriverType::DDSTYPE;
    }

    std::string GetMethodInfo() const noexcept
    {
        std::stringstream methodInfo;
        methodInfo << "serviceId=" << GetServiceId() << ", instanceId=" << GetInstanceId() << ", entityId=" <<
            GetEntityId() << ", requestTopicName=" << GetRequestTopicName() << ", replyTopicName=" <<
            GetReplyTopicName();
        return methodInfo.str();
    }
private:
    std::string requestTopicName_ {""};
    std::string replyTopicName_ {""};
    std::string qosProfile_ {vrtf::vcc::api::types::UNDEFINED_QOS_PROFILE};
    DomainId domainId_ {0U};
    uint32_t methodNum_ {1U};
    qos::ParticipantQos participantQos_ {qos::ParticipantQos(qos::DiscoveryFilter(0, "UNDEFINED_DISCOVERY_FILTER"))};
};
}
}
}

#endif
